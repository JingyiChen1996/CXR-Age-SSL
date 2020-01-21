import argparse
import cv2
import numpy as np
import copy
import torch
import imageio
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.cm as mpl_color_map

import os
import torch
import torchvision.transforms as tfms
from torch.autograd import Function
from torchvision import models
from dnn121 import DenseNet121, MobileNet

from segmentaion import get_mask, segment

class CamExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.model._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(-1, 1024)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x 


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
    
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input, target_class=None):
        
        conv_output, model_output = self.extractor.forward_pass(input)
        print('conv output:', conv_output.shape)
        print('model output:', model_output)
        if target_class == None:
            target_class = np.argmax(model_output.data.numpy())
        print('--------target-class:', target_class)

        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        
        self.model.model.zero_grad()
        self.model.fc.zero_grad()

        model_output.backward(gradient=one_hot_output, retain_graph=True)

        guided_gradients = self.extractor.gradients.data.numpy()[0]
        print("guided gradients:", guided_gradients.shape)
        target = conv_output.data.numpy()[0]
        print('target:', target.shape)
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2)) 

        #weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((256,256), Image.ANTIALIAS))/255
        return cam

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('cam_results'):
        os.makedirs('cam_results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')
    # Save colored heatmap
    path_to_file = os.path.join('cam_results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('cam_results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('cam_results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='/home/jingyi/cxr-jingyi/data/nih/v1/images/images_005/patient11178/study003/00011178_003.jpg',
                        help='Input image path')
    parser.add_argument('--save-path', type=str, default='00011178_003',
                        help='Image save parent path')
    parser.add_argument('--target-layer', type=int, default=13,
                        help='Target layer')                   

    args = parser.parse_args()

    return args
    


if __name__ == '__main__':
   
    args = get_args()
    random.seed(1) 
    torch.manual_seed(1) 
    torch.backends.cudnn.deterministic = True
    model = MobileNet(16)
    checkpoint = torch.load('/home/jingyi/cxr-jingyi/Age/result/supervised/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    grad_cam = GradCam(model=model, target_layer=args.target_layer)

    img = imageio.imread(args.image_path)
    cxr_test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512,512), interpolation=3),
    tfms.CenterCrop(256),
    tfms.ToTensor()
    ])
    img_mask = get_mask(img)
    cropped_img = segment(img, img_mask)
    # transformation
    preprocessed_img = cxr_test_transforms(cropped_img)
    #preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    print('input shape:', input.shape)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam.generate_cam(input, target_index)

    img_3d = cv2.cvtColor(cv2.resize(cropped_img,(256,256)), cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_3d)
    save_class_activation_images(img_pil, mask, args.save_path)
    print('Grad cam completed')
