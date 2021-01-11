# import os
# import random
# import sys
# import json

import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
# import torchvision.transforms as T

class COCOImage(Dataset):
    def __init__(self, anno_json, img_path, image_size):
        # anno_json : path/coco/annotations/*.json
        # img_path : path/coco/images/ directory
        self.anno = COCO(anno_json)
        self.img_path = img_path
        self.img_ids = self.anno.getImgIds()
        self.len = len(self.img_ids)
        self.image_size = image_size
    
    def __getitem__(self, idx):
        img_dict = self.anno.loadImgs(self.img_ids[idx])[0]
        file_name = img_dict['file_name']
        img_id = img_dict['id']

        img = cv2.imread(self.img_path + file_name)
        H0, W0 = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)

        return img, img_id, (H0, W0)
    
    def get_img(self, idx):
        img_dict = self.anno.loadImgs(idx)[0]
        file_name = img_dict['file_name']

        img = cv2.imread(self.img_path + file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __len__(self):
        return self.len
