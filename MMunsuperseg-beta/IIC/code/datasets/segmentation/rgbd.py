from __future__ import print_function

import os
import os.path as osp

import cv2
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils import data
from skimage import io
from tqdm import tqdm

from ...utils.segmentation.render import render
from ...utils.segmentation.transforms import \
  pad_and_or_crop, random_affine, custom_greyscale_numpy

__all__ = ["RGBD"]

RENDER_DATA = False


class _RGBD(data.Dataset):
  """Base class
  This contains fields and methods common to all Potsdam datasets:
  PotsdamFull (6)
  PotsdamFew (3)

  """

  def __init__(self, config=None, split=None, purpose=None, preload=False, include_height=True):
    super(_RGBD, self).__init__()

    self.include_height = include_height

    self.split = split
    self.purpose = purpose

    self.root = config.dataset_root

    self.single_mode = hasattr(config, "single_mode") and config.single_mode

    self.CLASS_DICT = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    # always used (labels fields used to make relevancy mask for train)
    self.gt_k = config.gt_k
    self.pre_scale_all = config.pre_scale_all
    self.pre_scale_factor = config.pre_scale_factor
    self.input_sz = config.input_sz

    self.include_rgb = config.include_rgb
    self.no_sobel = config.no_sobel

    # only used if purpose is train
    if purpose == "train":
      self.use_random_scale = config.use_random_scale
      if self.use_random_scale:
        self.scale_max = config.scale_max
        self.scale_min = config.scale_min

      self.jitter_tf = tvt.ColorJitter(brightness=config.jitter_brightness,
                                       contrast=config.jitter_contrast,
                                       saturation=config.jitter_saturation,
                                       hue=config.jitter_hue)

      self.flip_p = config.flip_p  # 0.5

      self.use_random_affine = config.use_random_affine
      if self.use_random_affine:
        self.aff_min_rot = config.aff_min_rot
        self.aff_max_rot = config.aff_max_rot
        self.aff_min_shear = config.aff_min_shear
        self.aff_max_shear = config.aff_max_shear
        self.aff_min_scale = config.aff_min_scale
        self.aff_max_scale = config.aff_max_scale

    self.preload = preload

    self.files = []
    self.images = []
    self.labels = []

    self._set_files()

    if self.preload:
      self._preload_data()

    cv2.setNumThreads(0)

  def _set_files(self):
    raise NotImplementedError()

  def _load_data(self, image_id):
    raise NotImplementedError()

  def _prepare_train(self, index, img, height):
    # This returns gpu tensors.
    # label is passed in canonical [0 ... 181] indexing

    img = img.astype(np.float32)
    height = height.astype(np.float32)
    height[np.isnan(height)] = 4.79

    # shrink original images, for memory purposes
    # or enlarge
    if self.pre_scale_all:
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      height = cv2.resize(height, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)      

    # basic augmentation transforms for both img1 and img2
    if self.use_random_scale:
      # bilinear interp requires float img
      scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                     self.scale_min
      img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      height = cv2.resize(height, dsize=None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)                       

    # random crop to input sz
    img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
    height, coords = pad_and_or_crop(height, self.input_sz, mode="random")

    # make img2 different from img1 (img)

    # tf_mat can be:
    # *A, from img2 to img1 (will be applied to img2's heatmap)-> img1 space
    #   input img1 tf: *tf.functional or pil.image
    #   input mask tf: *none
    #   output heatmap: *tf.functional (parallel), inverse of what is used
    #     for inputs, create inverse of this tf in [-1, 1] format

    # B, from img1 to img2 (will be applied to img1's heatmap)-> img2 space
    #   input img1 tf: pil.image
    #   input mask tf: pil.image (discrete)
    #   output heatmap: tf.functional, create copy of this tf in [-1,1] format

    # tf.function tf_mat: translation is opposite to what we'd expect (+ve 1
    # is shift half towards left)
    # but rotation is correct (-sin in top right = counter clockwise)

    # flip is [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # img2 = flip(affine1_to_2(img1))
    # => img1_space = affine1_to_2^-1(flip^-1(img2_space))
    #               = affine2_to_1(flip^-1(img2_space))
    # so tf_mat_img2_to_1 = affine2_to_1 * flip^-1 (order matters as not diag)
    # flip^-1 = flip

    # no need to tf label, as we're doing option A, mask needed in img1 space
    # converting to PIL does not change underlying np datatype it seems

    # images are RGBIR. We don't want to jitter or greyscale the IR part

    img1 = Image.fromarray(img.astype(np.uint8))

    # (img2) do jitter, no tf_mat change
    img2 = self.jitter_tf(img1)  # not in place, new memory
    img1 = np.array(img1)
    img2 = np.array(img2)
    height1 = height.copy()
    height2 = height.copy()

    # channels still last
    # channels still last
    if not self.no_sobel:
      img1 = custom_greyscale_numpy(img1, include_rgb=self.include_rgb)
      img2 = custom_greyscale_numpy(img2, include_rgb=self.include_rgb)

    img1 = img1.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.
    img1 = torch.from_numpy(img1).permute(2, 0 ,1)
    img2 = torch.from_numpy(img2).permute(2, 0, 1)
    
    height1 = height1 / 200.0
    height2 = height2 / 200.0
    height1 = torch.from_numpy(height1).unsqueeze(-1).permute(2, 0 ,1)
    height2 = torch.from_numpy(height2).unsqueeze(-1).permute(2, 0, 1)
    
    # (img2) do affine if nec, tf_mat changes
    if self.use_random_affine:
      affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                       "min_shear": self.aff_min_shear,
                       "max_shear": self.aff_max_shear,
                       "min_scale": self.aff_min_scale,
                       "max_scale": self.aff_max_scale}
      img2, affine1_to_2, affine2_to_1 = random_affine(img2,
                                                       **affine_kwargs)  ## tensors
      height2, _, _ = random_affine(height2,
                                                       **affine_kwargs)  ## tensors                                                       
                                                       
    else:
      affine2_to_1 = torch.zeros([2, 3]).to(torch.float32)  # identity
      affine2_to_1[0, 0] = 1
      affine2_to_1[1, 1] = 1

    # (img2) do random flip, tf_mat changes
    if np.random.rand() > self.flip_p:
      img2 = torch.flip(img2, dims=[2])  # horizontal, along width
      height2 = torch.flip(height2, dims=[2])  # horizontal, along width

      # applied affine, then flip, new = flip * affine * coord
      # (flip * affine)^-1 is just flip^-1 * affine^-1.
      # No order swap, unlike functions...
      # hence top row is negated
      affine2_to_1[0, :] *= -1.

    # uint8 tensor as masks should be binary, also for consistency,
    # but converted to float32 in main loop because is used
    # multiplicatively in loss
    mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8)

    if RENDER_DATA:
      render(img1, mode="image", name=("train_data_img1_%d" % index))
      render(img2, mode="image", name=("train_data_img2_%d" % index))
      render(height1, mode="height", name=("train_data_height1_%d" % index))
      render(height2, mode="height", name=("train_data_height2_%d" % index))
      render(affine2_to_1, mode="matrix",
             name=("train_data_affine2to1_%d" % index))
      render(mask_img1, mode="mask", name=("train_data_mask_%d" % index))

    img1 = torch.cat((img1,height1),0)
    img2 = torch.cat((img2,height2),0)
    #print(img1.shape, img2.shape)

    return img1, img2, affine2_to_1, mask_img1

  def _prepare_train_single(self, index, img, height):
    # Returns one pair only, i.e. without transformed second image.
    # Used for standard CNN training (baselines).
    # This returns gpu tensors.
    # label is passed in canonical [0 ... 181] indexing

    img = img.astype(np.float32)
    height = height.astype(np.float32)
    height[np.isnan(height)] = 4.79
    # shrink original images, for memory purposes
    # or enlarge
    if self.pre_scale_all:
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      height = cv2.resize(height, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)                       

    # basic augmentation transforms for both img1 and img2
    if self.use_random_scale:
      # bilinear interp requires float img
      scale_factor = (np.random.rand() * (self.scale_max - self.scale_min)) + \
                     self.scale_min
      img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      height = cv2.resize(height, dsize=None, fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_LINEAR)
    # random crop to input sz
    img, coords = pad_and_or_crop(img, self.input_sz, mode="random")
    height, coords = pad_and_or_crop(height, self.input_sz, mode="random")

    # converting to PIL does not change underlying np datatype it seems

    img1 = Image.fromarray(img.astype(np.uint8))

    img1 = self.jitter_tf(img1)  # not in place, new memory
    img1 = np.array(img1)
    height1 = height.copy()
    
    # channels still last
    # channels still last
    if not self.no_sobel:
      img1 = custom_greyscale_numpy(img1, include_rgb=self.include_rgb)

    img1 = img1.astype(np.float32) / 255.

    # convert both to channel-first tensor format
    # make them all cuda tensors now, except label, for optimality
    img1 = torch.from_numpy(img1).permute(2, 0, 1)

    height1 = height1 / 200.0
    height1 = torch.from_numpy(height1).unsqueeze(-1).permute(2, 0 ,1)


    if self.use_random_affine:
      affine_kwargs = {"min_rot": self.aff_min_rot, "max_rot": self.aff_max_rot,
                       "min_shear": self.aff_min_shear,
                       "max_shear": self.aff_max_shear,
                       "min_scale": self.aff_min_scale,
                       "max_scale": self.aff_max_scale}
      img1, _, _ = random_affine(img1, **affine_kwargs)  # tensors
      height1, _, _ = random_affine(height1, **affine_kwargs)  # tensors

    # (img2) do random flip, tf_mat changes
    if np.random.rand() > self.flip_p:
      img1 = torch.flip(img1, dims=[2])  # horizontal, along width
      height1 = torch.flip(height1, dims=[2])  # horizontal, along width

    # uint8 tensor as masks should be binary, also for consistency,
    # but converted to float32 in main loop because is used
    # multiplicatively in loss
    mask_img1 = torch.ones(self.input_sz, self.input_sz).to(torch.uint8)

    if RENDER_DATA:
      render(img1, mode="image", name=("train_data_img1_%d" % index))
      render(height1, mode="height", name=("train_data_height1_%d" % index))
      render(mask_img1, mode="mask", name=("train_data_mask_%d" % index))
    print(img1.shape, mask_img1.shape)
    
    img1 = torch.cat((img1,height1),0)
    return img1, mask_img1

  def _prepare_test(self, index, img, height, label):
    # This returns cpu tensors.
    #   Image: 3D with channels last, float32, in range [0, 1] (normally done
    #     by ToTensor).
    #   Label map: 2D, flat int64, [0 ... self.gt_k - 1]
    # label is passed in canonical [0 ... 181] indexing

    assert (label is not None)

    assert (img.shape[:2] == label.shape)
    img = img.astype(np.float32)
    label = label.astype(np.int32)
    height = height.astype(np.float32)
    height[np.isnan(height)] = 4.79
    
    # shrink original images, for memory purposes, or magnify
    if self.pre_scale_all:
      img = cv2.resize(img, dsize=None, fx=self.pre_scale_factor,
                       fy=self.pre_scale_factor,
                       interpolation=cv2.INTER_LINEAR)
      label = cv2.resize(label, dsize=None, fx=self.pre_scale_factor,
                         fy=self.pre_scale_factor,
                         interpolation=cv2.INTER_NEAREST)
      height = cv2.resize(height, dsize=None, fx=self.pre_scale_factor,
                         fy=self.pre_scale_factor,
                         interpolation=cv2.INTER_NEAREST)                         

    # center crop to input sz
    img, _ = pad_and_or_crop(img, self.input_sz, mode="centre")
    label, _ = pad_and_or_crop(label, self.input_sz, mode="centre")
    height, _ = pad_and_or_crop(height, self.input_sz, mode="centre")
    
    # finish
    # may be concatenating onto just greyscale image
    if not self.no_sobel:
      img = custom_greyscale_numpy(img, include_rgb=self.include_rgb)

    img = img.astype(np.float32) / 255.

    img = torch.from_numpy(img).permute(2, 0, 1)
    
    height = height / 200.0
    height = torch.from_numpy(height).unsqueeze(-1).permute(2, 0 ,1)
    

    if RENDER_DATA:
      render(label, mode="label", name=("test_data_label_pre_%d" % index))

    # convert to coarse if required, reindex to [0, gt_k -1], and get mask
    label = self._filter_label(label)
    mask = torch.ones(self.input_sz, self.input_sz).to(torch.uint8)

    if RENDER_DATA:
      render(img, mode="image", name=("test_data_img_%d" % index))
      render(label, mode="label", name=("test_data_label_post_%d" % index))
      render(mask, mode="mask", name=("test_data_mask_%d" % index))
      render(height, mode="height", name=("test_data_height_%d" % index))

    # dataloader must return tensors (conversion forced in their code anyway)
    img = torch.cat((img,height),0)
    return img, torch.from_numpy(label), mask

  def _preload_data(self):
    for image_id in tqdm(
      self.files, desc="Preloading...", leave=False, dynamic_ncols=True):
      image, label = self._load_data(image_id)
      self.images.append(image)
      self.labels.append(label)

  def __getitem__(self, index):
    if self.preload:
      image, label = self.images[index], self.labels[index]
    else:
      image_id = self.files[index]
      image, height, label = self._load_data(image_id)
    if self.purpose == "train":
      if not self.single_mode:
        return self._prepare_train(index, image, height)
      else:
        return self._prepare_train_single(index, image, height)
    else:
      assert (self.purpose == "test")
      return self._prepare_test(index, image, height, label)

  def __len__(self):
    return len(self.files)

  def _check_gt_k(self):
    raise NotImplementedError()

  def _filter_label(self, label):
    raise NotImplementedError()

  def _set_files(self):
    if self.split in ["unlabelled_train", "labelled_train", "labelled_test"]:
      if self.split.endswith('_train'):
        self.mode = 'training'
        file_list = os.listdir(osp.join(self.root, 'images', self.mode))
      else:
        self.mode = 'testing'
        file_list = os.listdir(osp.join(self.root, 'images', self.mode))
      self.files = file_list  # list of ids which may or may not have gt
    else:
      raise ValueError("Invalid split name: {}".format(self.split))

  def _load_data(self, image_id):
    image_path = osp.join(self.root, "images", self.mode, image_id)
    label_path = osp.join(self.root, "classes", self.mode, image_id.replace('RGB', 'CLS'))
    image = io.imread(image_path).astype(np.uint8)
    assert (image.dtype == np.uint8)
    if self.include_height:
        height_path = osp.join(self.root, "heights", self.mode, image_id.replace('RGB', 'AGL'))
        height = io.imread(height_path).astype(np.float32)
    else:
        height = None
        
    if os.path.exists(label_path):
      label = io.imread(label_path).astype(np.int32)
      assert (label.dtype == np.int32)
      for key, value  in self.CLASS_DICT.items():
        label[label == key] = value
      return image, height, label
    else:
      return image, height, None


class RGBD(_RGBD):
  def __init__(self, **kwargs):
    super(RGBD, self).__init__(**kwargs)

    config = kwargs["config"]

  def _filter_label(self, label):
      return label
