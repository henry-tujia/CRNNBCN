import argparse
import time
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info
import dataset
import pandas
#from warpctc_pytorch import CTCLoss

# from tensorboardX import SummaryWriter

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', default="lib/config/OWN_config.yaml", type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config

def main():

    # load config
    config = parse_arg()

    # create output folder
    # output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # # writer dict
    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(1))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()
    # criterion = CTCLoss()
    last_epoch = config.TRAIN.BEGIN_EPOCH

    model_state_file = "/home/wuxingxing/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2023-01-01-21-55/checkpoints/checkpoint_17_acc_0.0000.pth"
    if model_state_file == '':
        print(" => no checkpoint found")
    checkpoint = torch.load(model_state_file, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        model.load_state_dict(checkpoint)

    root_path = "/home/wuxingxing/crnn.pytorch/data/evaluation"
    model_info(model)

    res_list = []
    
    
    for dataset_inner in ["CUTE80","IC13_857","IC15_1811","IIIT5k_3000","SVT","SVTP"]:

        val_dataset = dataset.lmdbDataset(
            root=os.path.join(root_path,dataset_inner), transform=dataset.resizeNormalize((160, 32)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=True, batch_size=config.TEST.BATCH_SIZE_PER_GPU, num_workers=int(config.WORKERS))

        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

        start = time.time()

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, 0, None, None)

        time_diff = (time.time()-start)
        # is_best = acc > best_acc
        # best_acc = max(acc, best_acc)
        res_list.append({"CKPT":model_state_file,"Dataset":dataset_inner,"Acc":acc,"Cost":time_diff})
        # temp_series  = pandas.Series({"CKPT":model_state_file,"Dataset":dataset_inner,"Acc":0,"Cost":time_diff})
        # res_df = pandas.concat([res_df,pandas.DataFrame(temp_series)], axis=0)
    res_df = pandas.DataFrame(res_list)
    print(res_df)
    res_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(model_state_file)),"log",os.path.splitext(os.path.basename(model_state_file))[0]+".csv"))
        # print("Dataset:{}\tBest acc is:{}\tCost(s):{}".format(dataset_inner,best_acc,time_diff))
    # writer_dict['writer'].close()

if __name__ == '__main__':

    main()
