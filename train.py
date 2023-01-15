import argparse
import os

import horovod.torch as hvd
import torch
import torch.backends.cudnn as cudnn
import torch.nn
import yaml
from easydict import EasyDict as edict

import dataset
import lib.config.alphabets as alphabets
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.core import function
from lib.utils.utils import Metric, SmoothCTCLoss, model_info

# from lib.utils import SmoothCTCLoss
#from warpctc_pytorch import CTCLoss

# from tensorboardX import SummaryWriter



def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', default="lib/config/OWN_config.yaml", type=str)
    parser.add_argument("--local_rank", default=-1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.local_rank = int(args.local_rank)

    return config

def main():

    # load config
    config = parse_arg()
    hvd.init()

    if hvd.rank() == 0:
    # create output folder
        output_dict = utils.create_log_folder(config, phase='train')
    else:
        output_dict = None

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

    # # writer dict
    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    torch.cuda.set_device(hvd.local_rank())

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(4)
    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss().to(device)
    criterion_l = torch.nn.CrossEntropyLoss().to(device)
    # criterion = SmoothCTCLoss(config.MODEL.NUM_CLASSES + 1)
    last_epoch = config.TRAIN.BEGIN_EPOCH

    optimizer = utils.get_optimizer(config, model)

    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=hvd.Compression.none,
        backward_passes_per_step=1,
        op=hvd.Average,
        gradient_predivide_factor=1)



    # optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    train_dataset = dataset.lmdbDataset(root=config.DATASET.trainroot)
    assert train_dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False, sampler=train_sampler,
        num_workers=int(config.WORKERS),
        collate_fn=dataset.alignCollate(imgH=config.MODEL.IMAGE_SIZE.H, imgW=config.MODEL.IMAGE_SIZE.W, keep_ratio=True))
    

    val_dataset = dataset.lmdbDataset(
        root=config.DATASET.valroot, transform=dataset.resizeNormalize((100, 32)))
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False,sampler=None,batch_size=config.TEST.BATCH_SIZE_PER_GPU, num_workers=int(config.WORKERS))
    
        # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, None, output_dict,train_sampler,criterion_l)
        lr_scheduler.step()

        if hvd.rank() == 0:
            acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, None, output_dict)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            print("is best:", is_best)
            print("best acc is:", best_acc)
            # save checkpoint
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    # "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "best_acc": best_acc,
                },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            )

    # writer_dict['writer'].close()

if __name__ == '__main__':

    main()
