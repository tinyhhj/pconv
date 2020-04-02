"""
https://github.com/seoungwugoh/opn-demo
gated conv 2d
"""

import torch

def init_He(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

class GatedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.input_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.gating_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.activation = activation
        init_He(self)

    def forward(self, input):
        # O = act(Feature) * sig(Gating)
        feature = self.input_conv(input)
        if self.activation:
            feature = self.activation(feature)
        gating = self.sigmoid(self.gating_conv(input))
        return feature * gating

class ResidualBlock(torch.nn.Module):
    def __init__(self,dim,kernel_size=3,stride=1,padding=1,dilation=1):
        super().__init__()
        self.relu = torch.nn.ReLU(True)
        self.conv1 = torch.nn.Conv2d(dim, dim, kernel_size,stride,padding,dilation)
        self.conv2 = torch.nn.Conv2d(dim,dim, kernel_size, stride,padding,dilation)
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return out + x

