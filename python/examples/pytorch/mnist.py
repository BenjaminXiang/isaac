import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import isaac.pytorch
import os
import math


# Module
class ConvNet(nn.Module):
    def __init__(self, with_isaac=False):
        super(ConvNet, self).__init__()
        self.vgg1 = isaac.pytorch.VggBlock(1, 20, (3, 3), (2, 2), 'relu', 0.0005, pool=True, with_isaac=with_isaac, return_tmp=False)
        self.vgg2 = isaac.pytorch.VggBlock(20, 50, (3, 3), (2, 2), 'relu', 0.0005, pool=True, with_isaac=with_isaac, return_tmp=False)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        x = self.vgg1(x)
        x = self.vgg2(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, self.loss(x, target)
    
    def fuse(self):
        result = ConvNet(with_isaac=True)
        result.cuda()
        def copy(x, y):
            x.weight.data = y.weight.data.permute(1, 2, 3, 0).clone()
            x.bias.data = y.bias.data
        # Copy weights
        copy(result.vgg1.conv1, self.vgg1.conv1[0])
        copy(result.vgg1.conv2, self.vgg1.conv2[0])
        copy(result.vgg2.conv1, self.vgg2.conv1[0])
        copy(result.vgg2.conv2, self.vgg2.conv2[0])
        result.fc1.weight.data = self.fc1.weight.data.clone()
        result.fc1.bias.data = self.fc1.bias.data.clone()
        result.fc2.weight.data = self.fc2.weight.data.clone()
        result.fc2.bias.data = self.fc2.bias.data.clone()
        return result
    
    def quantize(self, x, target):
        history = dict()
        self.vgg1.arm_quantization(history)
        self.vgg2.arm_quantization(history)
        self.vgg2.pool.quantizer = None
        self.forward(x, target)
        return self


# Data Set
root, download = './data', True
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

# Data Loader
batch_size, opts = 128, {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, **opts)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, **opts)
ntrain, ntest = len(train_loader), len(test_loader)
        
# Training
model = ConvNet().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in range(5):
    # Update parameters
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
        _, train_loss = model(x, target)
        train_loss.backward()
        optimizer.step()
    # Evaluate validation error
    accuracy, test_loss = 0, 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
        score, loss = model(x, target)
        _, pred_label = torch.max(score.data, 1)
        accuracy += (pred_label == target.data).sum()
        test_loss += loss.data[0]
    accuracy /= ntest*batch_size
    test_loss /= ntest
    print('[Epoch {}] Train-loss: {:.4f} | Test-loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, train_loss.data[0], test_loss, accuracy))

# Quantization
x, target = next(iter(train_loader))
model = model.fuse().quantize(Variable(x.cuda()), Variable(target.cuda()))

# Inference
accuracy, test_loss = 0, 0
for batch_idx, (x, target) in enumerate(test_loader):
    x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
    score, loss = model(x, target)
    _, pred_label = torch.max(score.data, 1)
    accuracy += (pred_label == target.data).sum()
    test_loss += loss.data[0]
accuracy /= ntest*batch_size
print('Accuracy: {}'.format(accuracy))
