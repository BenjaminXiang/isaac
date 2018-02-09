from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from em.model.unet import unet3D_m1
import numpy as np
import copy
import isaac.pytorch.models
from time import time
import timeit
import h5py
from builtins import object
import argparse

def convert(legacy):
    result = isaac.pytorch.models.UNet().cuda()

    # Reorder indices because new state dict has upsample-upconv interleaved
    depth = legacy.depth
    ndown = 4*(depth + 1)
    reorder = list(range(ndown))
    for i in range(depth):
        upsamples = list(range(ndown + i*3 + 1, ndown + i*3 + 3))
        upconvs = list(range(ndown + depth*3 + i*4, ndown + depth*3 + i*4 + 4))
        reorder +=  upsamples + upconvs
    reorder += [ndown + 7*depth, ndown + 7*depth + 1]

    # Copy in proper order
    legacy_keys = list(legacy.state_dict().keys())
    result_keys = list(result.state_dict().keys())
    legacy_dict = legacy.state_dict()
    result_dict = result.state_dict()

    for i, j in enumerate(reorder):
        weights = legacy_dict[legacy_keys[j]].clone()
        # Transpose weights if necessary
        if(len(weights.size()) > 1):
            weights = weights.permute(1, 2, 3, 4, 0)
        # Copy weights
        result_dict[result_keys[i]] = weights
    result.load_state_dict(result_dict)

    return result


class DataIterator(object):

    def __init__(self, tile, data):
        self.current = [0, 0, 0]
        self.tile = tile
        self.sizes = data.shape
        self.data = torch.Tensor(data.reshape(1, 1, *self.sizes)).cuda()

    def __iter__(self):
        return self

    def __next__(self):
        i = np.random.randint(0, self.sizes[0] - self.tile[0])
        j = np.random.randint(0, self.sizes[1] - self.tile[1])
        k = np.random.randint(0, self.sizes[2] - self.tile[2])
        result = self.data[:, :, i:i+self.tile[0], j:j+self.tile[1], k:k+self.tile[2]]
        return result.clone(),


if __name__ == '__main__':
    # Program options
    parser = argparse.ArgumentParser(description='ISAAC Electron Microscopy Inference')
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('weights', help='path to model weights')
    parser.add_argument('--arch', '-a', default='unet', choices=['unet'])
    parser.add_argument('--batch-size', '-b', default=16, type=int, metavar='N', help='mini-batch size [default: 16]')
    parser.add_argument('--calibration-batches', '-c', default=4, type=int, metavar='N', help='number of batches for calibration [default: 16]')
    args = parser.parse_args()

    # Fix random seeds (for reproducibility)
    np.random.seed(0)

    # Load data
    T = np.array(h5py.File(args.data, 'r')['main']).astype(np.float32)/255
    dataset = DataIterator((31, 204, 204), T)
    iterator = iter(dataset)

    # Build models
    unet_ref = unet3D_m1().cuda()
    unet_ref.load_state_dict(torch.load(args.weights)['state_dict'])

    # Quantize
    print('Quantizing... ', end='', flush=True)
    unet_sc = convert(unet_ref)
    isaac.pytorch.quantize(unet_sc, iterator, args.calibration_batches)
    print('')

    # Benchmark
    print('Performance: ', end='', flush=True)
    X = Variable(next(iterator)[0], volatile=True).cuda()
    y_sc = unet_sc(X)
    Nvoxels = np.prod(y_sc.size()[2:])
    t_sc = [x for x in timeit.repeat(lambda: (unet_sc(X), torch.cuda.synchronize()), repeat=10, number=1)]
    t_ref = [x for x in timeit.repeat(lambda: (unet_ref(X), torch.cuda.synchronize()), repeat=10, number=1)]
    print('{:.2f} Mvox/s (Isaac) ; {:.2f} Mvox/s (PyTorch)'.format(Nvoxels/min(t_sc)*1e-6, Nvoxels/min(t_ref)*1e-6))

    # Evaluate
    print('Error: ', end='', flush=True)
    errors = np.zeros(128)
    for n in range(errors.size):
        X = Variable(next(iterator)[0], volatile=True).cuda()
        y_ref = unet_ref(X)
        y_sc = unet_sc(X)
        errors[n] = torch.norm(y_ref - y_sc).data[0]/torch.norm(y_ref).data[0]
    print('{:.4f} [+- {:.4f}]'.format(np.mean(errors), np.std(errors)))


