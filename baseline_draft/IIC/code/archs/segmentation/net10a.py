import torch.nn as nn
import torch.nn.functional as F

from ..cluster.vgg import VGGTrunk, VGGNet
from .multimodal_df import MMDFeatureFusion

__all__ = ["SegmentationNet10a"]
import math


# From first iteration of code, based on VGG11:
# https://github.com/xu-ji/unsup/blob/master/mutual_information/networks
# /vggseg.py

class SegmentationNet10aTrunk(VGGTrunk):
  def __init__(self, config, cfg):
    super(SegmentationNet10aTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    assert (config.input_sz % 2 == 0)

    self.conv_size = 3
    self.pad = 1
    self.cfg = cfg
    #self.in_channels = config.in_channels if hasattr(config, 'in_channels') \
    #  else 3
    self.in_channels = 4 if config.dataset == 'RGBD' else 3

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)  # do not flatten
    #print(x)
    #y = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class SegmentationNet10aHead(nn.Module):
  def __init__(self, config, output_k, cfg):
    super(SegmentationNet10aHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.cfg = cfg
    num_features = self.cfg[-1][0]

    self.num_sub_heads = config.num_sub_heads

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Conv2d(num_features, output_k, kernel_size=1,
                stride=1, dilation=1, padding=1, bias=False),
      nn.Softmax2d()) for _ in range(self.num_sub_heads)])

    self.input_sz = config.input_sz

  def forward(self, x):
    results = []
    for i in range(self.num_sub_heads):
      x_i = self.heads[i](x)
      x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")
      results.append(x_i)

    return results


class RGBDecoder(nn.Module):
    def __init__(self):
        super(RGBDecoder, self).__init__()
        self.in_dim = 512
        self.conv1 = nn.Conv2d(self.in_dim, self.in_dim // 2, 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.in_dim//2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.maxpool(x)  #128
        #x = self.conv1(x)
        #x = self.relu(self.bn(x))
        #x = self.maxpool(x)  #64
        return x


class HeightDecoder(nn.Module):
    def __init__(self):
        super(HeightDecoder, self).__init__()
        self.in_dim = 512 
        self.conv1 = nn.Conv2d(self.in_dim, self.in_dim // 2, 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.in_dim//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(self.in_dim // 2, 1, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        x = self.maxpool(x)
        hf = x
        x = self.conv2(x)
        return x, hf


class SegmentationNet10a(VGGNet):
  cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1),
         (512, 2), (512, 2)]  # 30x30 recep field

  def __init__(self, config):
    super(SegmentationNet10a, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)

    self.head = SegmentationNet10aHead(config, output_k=config.output_k,
                                       cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x):
    x = self.trunk(x)
    x = self.head(x)
    return x
