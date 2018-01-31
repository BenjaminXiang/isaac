import argparse
import os
import time, timeit

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import isaac.pytorch.models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',  help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,  metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()

    # Build data-loader
    val_dir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.Resize(256),  transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    image_folder = datasets.ImageFolder(val_dir, transformations)
    val_loader = torch.utils.data.DataLoader(image_folder, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Build models
    model = models.__dict__[args.arch](pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    input, target = next(iter(val_loader))
    input = torch.autograd.Variable(input, volatile=True).cuda()
    resnet_ref = models.resnet152(pretrained=True).cuda()
    resnet_ref.eval()
    resnet_sc = isaac.pytorch.models.resnet152()
    resnet_sc.quantize(input, approximate=True)

    # Benchmark
    y_ref = resnet_ref(input)
    y_sc = resnet_sc(input)
    t_sc = [x for x in timeit.repeat(lambda: (resnet_sc(input), torch.cuda.synchronize()), repeat=4, number=1)]
    t_ref = [x for x in timeit.repeat(lambda: (resnet_ref(input), torch.cuda.synchronize()), repeat=4, number=1)]
    print('Performance: {:.2f} Image/s (Isaac) ; {:.2f} Image/s (PyTorch)'.format(input.size()[0]/min(t_sc), input.size()[0]/min(t_ref)))

    # Accuracy
    criterion = nn.CrossEntropyLoss().cuda()
    validate(val_loader, resnet_sc, criterion)



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
