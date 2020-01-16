import sys
import bisect
from pathlib import Path

import numpy as np
import imageio
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as tfms

from utils import logger
#from image_preprocess import get_mask, segment

CXR_BASE = Path("/home/jingyi/cxr-jingyi/data").resolve()
NIH_CXR_BASE = CXR_BASE.joinpath("nih/v1").resolve()

def train_val_split(file_path):

    df = pd.read_csv(str(file_path)).fillna(0)
    train_df, val_df = train_test_split(df,
                                  train_size=0.8,
                                  stratify=df[['age_cat']])
    
    train_df.to_csv('~/cxr-jingyi/Age/train_8000.csv', index=False)
    val_df.to_csv('~/cxr-jingyi/Age/val_2000.csv', index=False)
    
    print('Write to csv!')
    return len(train_df)


def _load_manifest(file_path):
    
    df = pd.read_csv(str(file_path)).fillna(0)
    df_tmp = df[['path','age']]
    entries = df_tmp

    return entries


cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512,512), interpolation=3),
    #tfms.Resize(300, Image.LANCZOS),
    tfms.RandomRotation((-10, 10)),
    tfms.RandomCrop((256, 256)),
    tfms.RandomHorizontalFlip(),
    tfms.RandomVerticalFlip(),
    tfms.ToTensor(),
    #tfms.Normalize((0.1307,), (0.3081,))
])

cxr_test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512,512), interpolation=3),
    #tfms.Resize(256, Image.LANCZOS),
    tfms.CenterCrop(256),
    tfms.ToTensor(),
    #tfms.Normalize((0.1307,), (0.3081,))
])

 
def get_image(img_path, transforms):
    image = imageio.imread(img_path[0])
    # image preprocess
    # img_mask = get_mask(image)
    # cropped_img = segment(image, img_mask)
    # transformation
    #image_tensor = transforms(cropped_img)
    image_tensor = transforms(image)
    # print(image_tensor.shape)
    return image_tensor


class CxrDataset(Dataset):

    transforms = cxr_train_transforms

    def __init__(self, base_path, manifest_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = _load_manifest(manifest_path)
        self.base_path = base_path
       
    def __getitem__(self, index):

        def get_entries(index):
            df = self.entries.loc[index]
            paths = [self.base_path.joinpath(x).resolve() for x in df[0].split(',')]
            label = df[1:].tolist()
            #label = [df[1]]
            return paths, label

        img_path, label = get_entries(index)
        image_tensor = get_image(img_path, CxrDataset.transforms)
        target_tensor = torch.FloatTensor(label)
       
        return image_tensor, target_tensor

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def train():
        CxrDataset.transforms = cxr_train_transforms

    @staticmethod
    def eval():
        CxrDataset.transforms = cxr_test_transforms
