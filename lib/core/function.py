from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import horovod.torch as hvd

import torch

def cer(r: list, h: list):
     """
     Calculation of CER with Levenshtein distance.
     """
     # initialisation
     import numpy
     d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
     d = d.reshape((len(r) + 1, len(h) + 1))
     for i in range(len(r) + 1):
         for j in range(len(h) + 1):
             if i == 0:
                 d[0][j] = j
             elif j == 0:
                 d[i][0] = i

     # computation
     for i in range(1, len(r) + 1):
         for j in range(1, len(h) + 1):
             if r[i - 1] == h[j - 1]:
                 d[i][j] = d[i - 1][j - 1]
             else:
                 substitution = d[i - 1][j - 1] + 1
                 insertion = d[i][j - 1] + 1
                 deletion = d[i - 1][j] + 1
                 d[i][j] = min(substitution, insertion, deletion)
     return d[len(r)][len(h)] , float(len(r))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch,writer_dict=None, output_dict=None,train_sampler = None,criterion_l = None):
    train_sampler.set_epoch(epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # model.to(device)
    model.train()

    end = time.time()
    for i, (inp, text) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        # labels = utils.get_batch_label(dataset, idx)
        inp = inp.to(device)

        # inference
        preds_c = model.crnn(inp).cpu()

        # compute loss
        batch_size = inp.size(0)
        text, length,_ = converter.encode(text)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds_c.size(0)] * batch_size) # timestep * batchsize
        loss = criterion(preds_c, text, preds_size, length)

        preds_l = model.bcn(preds_c.to(device),preds_size.to(device)).cpu()

        flat_pt_logits = utils._flatten(preds_l.permute(1,0,2), length)
        # preds_l_size = torch.IntTensor([preds_l.size(0)] * batch_size)
        loss_l = criterion_l(flat_pt_logits,text.long())
        # loss_l = criterion_l(preds_l, text)

        loss += 0.01*loss_l
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))
        batch_time.update(time.time()-end)

        _, preds = preds_c.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        sim_preds_encoded,_,_ = converter.encode(sim_preds)

        temp_correct, temp_sum = cer(text.tolist(),sim_preds_encoded.tolist())

        acc = (temp_sum-temp_correct)/temp_sum

        if hvd.rank() ==0:
            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                    'Acc {acc_temp:.5f}\t'\
                        .format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=inp.size(0)/batch_time.val,
                        data_time=data_time, loss=losses,acc_temp = acc)
                print(msg)

                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.avg, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.to(device)
    model.eval()

    n_correct = 0
    sum_all = 0
    with torch.no_grad():
        for i, (inp, text) in enumerate(val_loader):

            # labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)


            # inference
            preds_c = model.crnn(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length,raw_text = converter.encode(text)                   # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
            preds_size = torch.IntTensor([preds_c.size(0)] * batch_size) # timestep * batchsize
            loss = criterion(preds_c, text, preds_size, length)


            losses.update(loss.item(), inp.size(0))


            _, preds = preds_c.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            sim_preds_encoded,_,_ = converter.encode(sim_preds)

            temp_correct, temp_sum = cer(text.tolist(),sim_preds_encoded.tolist())
            n_correct += (temp_sum-temp_correct)
            sum_all += temp_sum

            if i == config.TEST.NUM_TEST_BATCH:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, raw_text):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    # num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    # if num_test_sample > len(dataset):
    #     num_test_sample = len(dataset)
    accuracy = n_correct / float(sum_all)

    print("[#correct:{} / #total:{}]".format(n_correct, sum_all))

    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy




