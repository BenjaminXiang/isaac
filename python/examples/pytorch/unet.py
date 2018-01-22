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


if __name__ == '__main__':
    # Load data
    T = np.array(h5py.File('./im_uint8_half.h5','r')['main']).astype(np.float32)/255
    I, J, K = 31, 204, 204

    # Build models
    unet_ref = unet3D_m1().cuda()
    unet_ref.load_state_dict(torch.load('./net_iter_100000_m1.pth')['state_dict'])

    # Quantize
    X = T[:I, :J, :K].reshape(1, 1, I, J, K)
    X = Variable(torch.from_numpy(X), volatile=True).cuda()
    unet_sc = convert(unet_ref).quantize(X, approximate=True)

    # Evaluate errors
    N = 10
    errors = np.zeros(N)
    np.random.seed(0)
    i_range = np.random.randint(I, T.shape[0] - I, N)
    j_range = np.random.randint(J, T.shape[1] - J, N)
    k_range = np.random.randint(K, T.shape[2] - K, N)
    for n, (i, j, k) in enumerate(zip(i_range, j_range, k_range)):
        X = T[i : i+I, j : j+J, k : k+K].reshape(1, 1, I, J, K)
        X = Variable(torch.from_numpy(X)).cuda()
        y_ref = unet_ref(X)
        y_sc = unet_sc(X)
        errors[n] = torch.norm(y_ref - y_sc).data[0]/torch.norm(y_ref).data[0]
    print('Error: {:.4f} [+- {:.4f}]'.format(np.mean(errors), np.std(errors)))

    # Benchmark
    t_sc = [int(x*1e3) for x in timeit.repeat(lambda: (unet_sc(X), torch.cuda.synchronize()), repeat=1, number=1)]
    t_ref = [int(x*1e3) for x in timeit.repeat(lambda: (unet_ref(X), torch.cuda.synchronize()), repeat=1, number=1)]
    print('Time: {}ms (Isaac) ; {}ms (PyTorch)'.format(t_sc[0], t_ref[0]))
