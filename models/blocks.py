import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Downblock(nn.Module):
    def __init__(self, channels, kernel_size=3, relu=True, stride=2, padding=1):
        super(Downblock, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu

    def forward(self, x):
        x = self.bn(self.dwconv(x))
        if self.relu: x = F.relu(x)
        return x

class GEBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, spatial, extent=0, extra_params=True, mlp=True, dropRate=0.0):
        # If extent is zero, assuming global.
        super(GEBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))

        self.conv2 = nn.Sequential(nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True), 
            nn.Dropout(p=dropRate), nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False))

        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, 
            out_planes, kernel_size=1, stride=stride, bias=False) or None

        if extra_params:
            if extent == 0:
                # Global DW Conv + BN
                self.downop = Downblock(out_planes, relu=False, kernel_size=spatial)
            elif extent == 2:
                self.downop = Downblock(out_planes, relu=False)
            elif extent == 4:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True), Downblock(out_planes, relu=False))
            elif extent == 8:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                    Downblock(out_planes, relu=True), Downblock(out_planes, relu=False))
            else:
                raise NotImplementedError('Extent must be 0,2,4 or 8 for now')
        else:
            self.downop = nn.AdaptiveAvgPool2d(spatial // extent) if extent else self.downop = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16, kernel_size=1, padding=0, bias=False), nn.ReLU(),
            nn.Conv2d(out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False)) if mlp else lambda x: x

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        # Down, up, sigmoid
        map = self.mlp(self.downop(out))
        # Assuming squares because lazy.
        map = F.interpolate(map, out.shape[-1])
        map = torch.sigmoid(map)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out * map)
