import os
import numpy as np
import time
import sys
import imageio
from PIL import Image
import bisect
from pathlib import Path

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as tfms

from dnn169 import DenseNet169

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, transCrop):
       
        #---- Initialize the network
        model = DenseNet169(True).cuda()
        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        #model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model.module.densenet169.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        cxr_test_transforms = tfms.Compose([
            tfms.ToPILImage(),
            tfms.Resize(256, Image.LANCZOS),
            tfms.CenterCrop(256),
            tfms.ToTensor(),
            #tfms.Normalize((0.1307,), (0.3081,))
        ])
   
        
        self.transformSequence = cxr_test_transforms
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        image = imageio.imread(pathImageFile)
        image_tensor = self.transformSequence(image)
        #imageData = Image.open(pathImageFile).convert('RGB')
        #imageData = self.transformSequence(imageData)
        imageData = image_tensor.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(str(pathImageFile), 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 
CXR_BASE = Path("../cxr").resolve()
STANFORD_CXR_BASE = CXR_BASE.joinpath("stanford/v1").resolve()
pathInputImage = STANFORD_CXR_BASE.joinpath('train/patient09533/study1/view1_frontal.jpg').resolve()
pathOutputImage = 'heatmap.png'
pathModel = 'runtime/age-stanford-female/model_epoch_100.0.pth.tar'

transCrop = 224

h = HeatmapGenerator(pathModel, transCrop)
h.generate(pathInputImage, pathOutputImage, transCrop)


CXR_BASE = Path("/home/jingyi/cxr-jingyi/data").resolve()
NIH_CXR_BASE = CXR_BASE.joinpath("nih/v1").resolve()
test_df = pd.read_csv("~/cxr-jingyi/Age/NIH_test_2500.csv") 

path1 = test_df.iloc[0]['path']
path1 = NIH_CXR_BASE.joinpath(path1).resolve()

import matplotlib.pyplot as plt
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

image = load_image(str(path1))
plt.imshow(image)

model = MobileNet(16)
checkpoint = torch.load('/home/jingyi/cxr-jingyi/Age/result/supervised/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

backprop = Backprop(model)
# Transform the input image to a tensor

owl = apply_transforms(image)

# Set a target class from ImageNet task: 24 in case of great gray owl

target_class = 16

# Ready to roll!

backprop.visualize(owl, target_class, guided=True)




