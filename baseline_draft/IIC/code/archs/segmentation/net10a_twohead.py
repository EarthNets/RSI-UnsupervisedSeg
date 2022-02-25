from .net10a import SegmentationNet10aHead, SegmentationNet10aTrunk, \
  SegmentationNet10a, HeightDecoder, RGBDecoder
from ..cluster.vgg import VGGNet
import torch
from . import fpn, backbone
import torch.nn as nn 
import torch.nn.functional as F
import pdb
from .multimodal_df import MMDFeatureFusion


__all__ = ["SegmentationNet10aTwoHead", "SegmentationNetRN50TwoHead"]


class SegmentationNet10aTwoHead(VGGNet):
  def __init__(self, config):
    super(SegmentationNet10aTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = SegmentationNet10aTrunk(config, cfg=SegmentationNet10a.cfg)
    #self.rgb_decoder = RGBDecoder()
    #self.height_decoder = HeightDecoder()
    #self.conv1 = nn.Conv2d(128, 256, 3, 1, 1)
    #self.bn1 = nn.BatchNorm2d(256)
    #self.conv2 = nn.Conv2d(256, 512, 3, 1, 1)
    #self.bn2 = nn.BatchNorm2d(512)
    #self.relu = nn.ReLU()
    #self.mmfuse = MMDFeatureFusion(dim=256, fmap_size=64)
    self.head_A = SegmentationNet10aHead(config, output_k=config.output_k_A,
                                         cfg=SegmentationNet10a.cfg)
    self.head_B = SegmentationNet10aHead(config, output_k=config.output_k_B,
                                         cfg=SegmentationNet10a.cfg)

    self._initialize_weights()

  def forward(self, x, head="B"):
    x = self.trunk(x) #4,512,252,252
    #rgbf = self.rgb_decoder(x) #N, 512, 64, 64
    #hout, hf = self.height_decoder(x) #N, 512, 64, 64
    #x = self.mmfuse(rgbf, hf) #N, 128, 64, 64
    #x = F.interpolate(x, (128,128), mode='bilinear', align_corners=True)
    #x = self.relu(self.bn1(self.conv1(x)))
    #x = F.interpolate(x, (252,252), mode='bilinear', align_corners=True)
    #x = self.relu(self.bn2(self.conv2(x)))

    if head == "A":
      x = self.head_A(x)
    elif head == "B":
      x = self.head_B(x)
    else:
      assert (False)

    #return x, hout
    return x


class FPNDecoder(nn.Module):
    def __init__(self, config):
        super(FPNDecoder, self).__init__()

        mfactor = 4
        out_dim = 256

        self.layer4 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

class FPNHead(nn.Module):
  def __init__(self, config, output_k):
    super(FPNHead, self).__init__()

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



class SegmentationNetRN50TwoHead(torch.nn.Module):
    def __init__(self, config):
        super(SegmentationNetRN50TwoHead, self).__init__()
        
        self.backbone = backbone.__dict__['resnet50']()
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

