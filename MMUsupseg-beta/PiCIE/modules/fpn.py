import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from .swin_transformer import SwinTransformer


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        if 'swin' in args.arch:
            if args.rgbd:
                in_chans = 4
            else:
                in_chans = 3
            self.backbone = SwinTransformer(img_size=512,
                                            patch_size=4,
                                            in_chans=in_chans,
                                            num_classes=150,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=8,
                                            mlp_ratio=4,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0,
                                            drop_path_rate=0.3,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)
        elif 'resnet' in args.arch:
            self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        else:
            print('arch not defined.')
            raise NotImplementedError
            
        self.decoder  = FPNDecoder(args)

    def forward(self, x):
        feats = self.backbone(x)
        outs  = self.decoder(feats) 

        return outs 

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        self.arch = args.arch 
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
            o4 = self.upsample_add(o3, self.layer4(x['res2']))  # 128*128
        elif 'swin' in self.arch:
            o1 = self.layer1(x['swin3'])
            o2 = self.upsample_add(o1, self.layer2(x['swin2']))
            o3 = self.upsample_add(o2, self.layer3(x['swin1']))
            o4 = self.upsample_add(o3, self.layer4(x['swin0'])) # 64*64          
            # upsample to 128
            _, _, H, W = o4.size()
            o4 = F.interpolate(o4, size=(H*2, W*2), mode='bilinear', align_corners=False)
        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 



