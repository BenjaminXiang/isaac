import torch
import torch.nn as nn
import isaac.pytorch as sc
import torch.utils.model_zoo as model_zoo

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_num, out_num, stride = 1, downsample = None, dim=2):
        super(BasicBlock, self).__init__()
        self.conv1 =  sc.ConvType[dim](in_num, out_num, 3, bias = False, activation = 'relu', alpha = 0)
        self.conv2 =  sc.ConvType[dim](out_num, out_num, 3, bias = False, activation = 'relu', alpha = 0, residual = 'add')
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out, residual)
        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(Bottleneck, self).__init__()
        self.conv1 = sc.ConvType[dim](in_num, out_num, kernel_size=1, bias=False, activation='linear')
        self.conv2 = sc.ConvType[dim](out_num, out_num, kernel_size=3, stride=stride, padding=1, bias=False, activation='linear')
        self.conv3 = sc.ConvType[dim](out_num, out_num*4, kernel_size=1, bias=False, activation='relu', residual='add')
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out, residual)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_num = 64
        self.conv1 = sc.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, activation='relu', alpha=0)
        self.maxpool = sc.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        # Downsampling
        downsample = None
        if stride != 1 or self.in_num != planes * block.expansion:
            downsample = sc.Conv2d(self.in_num, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
        # Layers
        layers = [block(self.in_num, planes, stride, downsample)]
        self.in_num = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_num, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def convert(model, reference):
    reference_dict = reference
    batch_norm_keys = [x for x in reference_dict.keys() if any([s in x for s in ('downsample.1', 'bn')])]
    reference_keys = [x for x in reference_dict.keys() if x not in batch_norm_keys]
    result_dict = model.state_dict()
    result_keys = result_dict.keys()

    # Copy weights
    for i_key, j_key in zip(result_keys, reference_keys):
        weights = reference_dict[j_key].data.clone()
        # Transpose weights if necessary
        if(len(weights.size()) == 4):
            weights = weights.permute(1, 2, 3, 0)
        result_dict[i_key] = weights

    # Fold Batch Normalization
    conv_name = lambda x: x.replace('bn', 'conv') if 'bn' in x else x.replace('downsample.1', 'downsample')
    batch_norm_keys = list(set(['.'.join(x.split('.')[:-1]) for x in batch_norm_keys]))
    conv_keys = map(conv_name, batch_norm_keys)
    for x, y in zip(conv_keys, batch_norm_keys):
        eps = 1e-5
        # Extract scales
        scale_bias = reference_dict['{}.bias'.format(y)].data
        scale_weight = reference_dict['{}.weight'.format(y)].data
        mean = reference_dict['{}.running_mean'.format(y)]
        var = reference_dict['{}.running_var'.format(y)]
        alpha = scale_weight / (torch.sqrt(var) + eps)
        # Adjust conv weights/bias
        try:
            conv_bias = result_dict['{}.bias'.format(x)]
            conv_bias = conv_bias*alpha + (scale_bias - mean*alpha)
        except KeyError:
            pass
        conv_weight = result_dict['{}.weight'.format(x)]
        for i in range(len(alpha)):
            conv_weight[:,:,:,i] *= alpha[i]

    # Write back state dict
    model.load_state_dict(result_dict)



pretrained_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs).cuda()
    convert(model, model_zoo.load_url(pretrained_urls['resnet18']))
    return model
