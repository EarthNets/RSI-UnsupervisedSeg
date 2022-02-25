import os 
import torch 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
import json
import random 
import cv2
import pickle
import rasterio
import pdb

# We remove ignore classes. 
CLASS_DICT = {0:0, 2:0, 5:1, 6:2, 9:3, 17:4, 65:0}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class EvalRGBD(data.Dataset):
    def __init__(self, root, split, mode, res=1024, transform_list=[], label=True, label_mode='gtFine', long_image=False, include_height=False):
        self.root  = root 
        self.split = split
        self.mode  = mode
        self.res   = res 
        self.label = label

        self.label_mode = label_mode  
        self.long_image = long_image 
        self.include_height = include_height
        
        # Create label mapper.:
        self.CLASS_DICT = CLASS_DICT
        # For test-time augmentation / robustness test. 
        self.transform_list = transform_list

        self.imdb = self.load_imdb()
        
    def load_imdb(self):
        imdb = []

        if self.split=='val':
            folder = 'val'
        elif self.split=='train':
            folder = 'training'
        elif self.split=='test':
            folder = 'testing'
        
        
        for fname in os.listdir(os.path.join(self.root, 'images', folder)):
            image_path = os.path.join(self.root, 'images', folder, fname)
            ### height ###
            if self.include_height:
                height_path = os.path.join(self.root, 'heights', folder, fname.replace('RGB', 'AGL'))
            else:
                height_path = None
            ##############    
            if self.split != 'train':
                # First load fine-grained labels and map ourselves.
                lname = fname.replace('RGB', 'CLS')
                label_path = os.path.join(self.root, 'classes', folder, lname)
            else:
                label_path = None 

            imdb.append((image_path,height_path,label_path))
            
        torch.save(imdb, 'outputs/rgbd/RGBD_pathlist.pth')
        return imdb
        
    
    def __getitem__(self, index):
        impath, hpath, gtpath = self.imdb[index]

        image = Image.open(impath).convert('RGB')
        #height = Image.open(hpath) if self.include_height else None
        if self.include_height:
            height_np = rasterio.open(hpath).read(1)
            height_np[np.isnan(height_np)] = 4.79
            height = Image.fromarray(height_np)
        else:
            height = None        
        label = Image.open(gtpath) if self.label else None 
        return (index,) + self.transform_data(image, height, label, index)


    def transform_data(self, image, height, label, index):

        # 1. Resize
        image = TF.resize(image, self.res, Image.BILINEAR)
        
        # 2. CenterCrop
        if not self.long_image:
            w, h = image.size
            left = int(round((w - self.res) / 2.))
            top  = int(round((h - self.res) / 2.))

            image = TF.crop(image, top, left, self.res, self.res)
            
        # 3. Transformation
        image = self._image_transform(image, self.mode)
        #if not self.label:
        #    return (image, None)
        if self.include_height:
            height = TF.resize(height, self.res, Image.BILINEAR)
            height = TF.crop(height, top, left, self.res, self.res) if not self.long_image else height
            height = self._height_transform(height)
            #pdb.set_trace()
            ## concatenate rgb and dsm
            image = torch.cat((image,height),0)
            
        if self.label:
            label = TF.resize(label, self.res, Image.NEAREST)
            label = TF.crop(label, top, left, self.res, self.res) if not self.long_image else label
            label = self._label_transform(label)

        return image, label


    def _label_transform(self, label):
        label = np.array(label)
        #if self.dc:
        label = np.where(label>0, label-1, label)
            #for i in [2,3]:
            #    print(np.any(label==i))                            

        return torch.LongTensor(label)

    def _height_transform(self,height):
        trans_list = [transforms.ToTensor(), transforms.Normalize(mean=[4.79],std=[61.67])]
        return transforms.Compose(trans_list)(height)


    def _image_transform(self, image, mode):
        if self.mode == 'test':
            transform = self._get_data_transformation()

            return transform(image)
        else:
            raise NotImplementedError()


    def _get_data_transformation(self):
        trans_list = []
        if 'jitter' in self.transform_list:
            trans_list.append(transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8))
        if 'grey' in self.transform_list:
            trans_list.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in self.transform_list:
            trans_list.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5))
        
        # Base transformation
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        return transforms.Compose(trans_list)


    
    def __len__(self):
        return len(self.imdb)
        

  
            
       
