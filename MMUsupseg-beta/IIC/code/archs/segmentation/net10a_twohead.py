from .net10a import SegmentationNet10aHead, SegmentationNet10aTrunk, \
  SegmentationNet10a
from ..cluster.vgg import VGGNet
import torch
from . import fpn, backbone
import torch.nn as nn 
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
import pdb

__all__ = ["SegmentationNet10aTwoHead", "SegmentationNetRNTwoHead"]


class SegmentationNet10aTwoHead(VGGNet):
  def __init__(self, config):
    super(SegmentationNet10aTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)
    self.head_A = SegmentationNet10aHead(config, output_k=config.output_k_A,
                                         cfg=SegmentationNet10a.cfg)
    self.head_B = SegmentationNet10aHead(config, output_k=config.output_k_B,
                                         cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x, head="B"):
    x = self.trunk(x)
    if head == "A":
      x = self.head_A(x)
    elif head == "B":
      x = self.head_B(x)
    else:
      assert (False)

    return x


class FPNDecoder(nn.Module):
    def __init__(self, config):
        super(FPNDecoder, self).__init__()
        self.arch = config.backbone 
        if self.arch == 'resnet18':
            mfactor = 1
            out_dim = 128 
        elif self.arch == 'resnet50':
            mfactor = 4
            out_dim = 256
        elif self.arch == 'swin_t':
            mfactor = 1
            out_dim = 256
        else:
            print('arch not defined.')
            raise NotImplementedError        

        if 'resnet' in self.arch:
            self.layer4 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1, stride=1, padding=0)
            self.layer3 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
            self.layer2 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
            self.layer1 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1, stride=1, padding=0)
        elif 'swin' in self.arch:
            self.layer4 = nn.Conv2d(768*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
            self.layer3 = nn.Conv2d(768*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
            self.layer2 = nn.Conv2d(768*mfactor, out_dim, kernel_size=1, stride=1, padding=0)
            self.layer1 = nn.Conv2d(768*mfactor, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if 'resnet' in self.arch:
            o1 = self.layer1(x['res5'])
            o2 = self.upsample_add(o1, self.layer2(x['res4']))
            o3 = self.upsample_add(o2, self.layer3(x['res3']))
            o4 = self.upsample_add(o3, self.layer4(x['res2']))
        elif 'swin' in self.arch:
            o1 = self.layer1(x['swin3'])
            o2 = self.upsample_add(o1, self.layer2(x['swin2']))
            o3 = self.upsample_add(o2, self.layer3(x['swin1']))
            o4 = self.upsample_add(o3, self.layer4(x['swin0'])) 

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

class FPNHead(nn.Module):
  def __init__(self, config, output_k):
    super(FPNHead, self).__init__()
    if config.backbone=='resnet18':
        num_features = 128
    else:
        num_features = 256
    self.num_sub_heads = config.num_sub_heads

    self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(num_features, output_k, kernel_size=1, stride=1, dilation=1, padding=1, bias=False), nn.Softmax2d()) for _ in range(self.num_sub_heads)])

    self.input_sz = config.input_sz

  def forward(self, x):
    results = []
    for i in range(self.num_sub_heads):
      x_i = self.heads[i](x)
      if not x_i.shape[-1] == self.input_sz:
        x_i = F.interpolate(x_i, size=self.input_sz, mode="bilinear")
      results.append(x_i)

    return results



class SegmentationNetRNTwoHead(torch.nn.Module):
    def __init__(self, config):
        super(SegmentationNetRNTwoHead, self).__init__()
        
        self.backbone = backbone.__dict__[config.backbone]()
        self.decoder = FPNDecoder(config)
        if config.dataset=='RGBD':
            self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)        
        self.head_A = FPNHead(config, output_k=config.output_k_A)
        self.head_B = FPNHead(config, output_k=config.output_k_B)
        
        
    def forward(self, x, head="B"):
        x = self.backbone(x)
        #pdb.set_trace()
        x = self.decoder(x)
        
        if head == "A":
            x = self.head_A(x)
        elif head == "B":
            x = self.head_B(x)
        else:
            assert (False)

        return x 

