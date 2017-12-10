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

# Quantization
def quantize(input, bits):
    bound = math.pow(2.0, bits-1)
    min = - bound
    max = bound - 1
    scale = max / input.abs().max()
    rounded = torch.floor(input*scale + 0.5)
    return torch.clamp(rounded, min, max)/scale, scale
    
# Module
class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1_act = nn.Sequential(nn.Conv2d(1, 20, (5, 5)), nn.LeakyReLU(0.05))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_act = nn.Sequential(nn.Conv2d(20, 64, (5, 5)), nn.LeakyReLU(0.05))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*64, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()
        
    def forward(self, x, target):
        x = self.conv1_act(x)
        x = self.pool1(x)
        x = self.conv2_act(x)
        x = self.pool2(x)
        x = x.view(-1, 4*4*64)
        x = self.fc1(x)
        x = self.fc2(x)
        loss = self.ceriation(x, target)
        return x, loss
        
    def path(self):
        return 'network/conv2d.pth'


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
if not os.path.exists(model.path()):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(10):
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
    torch.save(model.state_dict(), model.path())

# Inference
#model.conv1_act = nn.Sequential(isaac.pytorch.Conv2d(1, 20, (5, 5), (1, 1), activation='relu', alpha=0.05))
#model.conv2_act = nn.Sequential(isaac.pytorch.Conv2d(20, 50, (5, 5), (1, 1), activation='relu', alpha=0.05))
model.load_state_dict(torch.load(model.path()))
model.conv1_act[0].weight.data = quantize(model.conv1_act[0].weight.data, 8)[0]
model.conv2_act[0].weight.data = quantize(model.conv2_act[0].weight.data, 8)[0]
model.fc1.weight.data = quantize(model.fc1.weight.data, 8)[0]
model.fc2.weight.data = quantize(model.fc2.weight.data, 8)[0]
#model.conv1_act[0].weight.data = model.conv1_act[0].weight.data.permute(1, 2, 3, 0)
#model.conv2_act[0].weight.data = model.conv2_act[0].weight.data.permute(1, 2, 3, 0)
accuracy, test_loss = 0, 0
for batch_idx, (x, target) in enumerate(test_loader):
    x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
    score, loss = model(x, target)
    _, pred_label = torch.max(score.data, 1)
    accuracy += (pred_label == target.data).sum()
    test_loss += loss.data[0]
accuracy /= ntest*batch_size
print('Accuracy: {}'.format(accuracy))
