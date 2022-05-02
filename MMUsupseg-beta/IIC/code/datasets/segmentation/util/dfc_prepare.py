import glob
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import scipy.io as sio
from skimage import io

# split each 6000x6000 image into 15x15 half scaled 200x200 images
# make train and test lists

SOURCE_IMGS_DIR = "/data/data/dfc/images/"
SOURCE_IMGS_SUFFIX = "_RGB.tif"
SOURCE_GT_DIR = "/data/data/dfc/classes/"
SOURCE_GT_SUFFIX = "_CLS.tif"

OUT_DIR = "/data/data/dfc/IIC/"


CLASS_DICT = {0: 0, 2: 1, 5: 2, 6: 3, 9: 4, 17: 5, 65: 0}

def main():
  out_dir_imgs = os.path.join(OUT_DIR, "imgs")
  out_dir_gt = os.path.join(OUT_DIR, "gt")

  if not os.path.exists(out_dir_imgs):
    os.makedirs(out_dir_imgs)
  if not os.path.exists(out_dir_gt):
    os.makedirs(out_dir_gt)
  
  unlabelled_train = open(os.path.join(OUT_DIR, "unlabelled_train.txt"), "w+")
  labelled_train = open(os.path.join(OUT_DIR, "labelled_train.txt"), "w+")
  labelled_test = open(os.path.join(OUT_DIR, "labelled_test.txt"), "w+")
  for mode in ['training', 'validation']:
    for i, img_path in enumerate(sorted(glob.glob(SOURCE_IMGS_DIR+ mode + "/*.tif"))):
      print("on img: %d %s" % (i, datetime.now()))

      handle = os.path.basename(img_path)[:-len(SOURCE_IMGS_SUFFIX)]
      img = io.imread(img_path)

      sio.savemat(os.path.join(out_dir_imgs, "%s.mat" % str(os.path.basename(img_path)[:-4])), {"img": img})
      if mode == 'training':
        unlabelled_train.write("%s\n" % os.path.basename(img_path)[:-4])
        labelled_train.write("%s\n" % os.path.basename(img_path)[:-4])
      else:
        labelled_test.write("%s\n" % os.path.basename(img_path)[:-4])

      gt_path = os.path.join(SOURCE_GT_DIR, mode, handle + SOURCE_GT_SUFFIX)
      gt = io.imread(gt_path)
      for key, value in CLASS_DICT.items():
        gt[gt==key] = value

      sio.savemat(os.path.join(out_dir_gt, "%s.mat" % str(os.path.basename(gt_path)[:-4])), {"gt": gt})

  unlabelled_train.close()
  labelled_train.close()
  labelled_test.close()


if __name__ == "__main__":
  main()
