from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import numpy as np
import torchvision.transforms as T
from torch import nn
class AugmentImage(nn.Module):
    def __init__(self):
        super(AugmentImage, self).__init__()
# 1. Simple transformations
# Resize
    def Resize(self, orig_img):
        resized_imgs = [T.Resize(size=size)(orig_img) for size in [32,128]]
        return resized_imgs
# Gray Scale
    def Grayscale(self, orig_img):
        return T.Grayscale()(orig_img)
# Normalize
    def Normalize(self, orig_img):
        normalized_img = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(orig_img)) 
        return [T.ToPILImage()(normalized_img)]
# Random Rotation
    def RandomRotation(self, orig_img):
        return [T.RandomRotation(degrees=d)(orig_img) for d in range(50,151,50)]
# Center Crop
    def CenterCrop(self, orig_img):
        return [T.CenterCrop(size=size)(orig_img) for size in (128, 64, 32)]
# Random Crop
    def RandomCrop(self, orig_img):
        return [T.RandomCrop(size=size)(orig_img) for size in (832,704, 256)]
# Gaussian Blur
    def GaussianBlur(self, orig_img):
        return [T.GaussianBlur(kernel_size=(51, 91), sigma=sigma)(orig_img) for sigma in (3,7)]
# 2. More advanced techniques
# Gaussian Noise
    def GaussianNoise(self, orig_img):        
        def add_noise(inputs,noise_factor=0.3):
            noisy = inputs+torch.randn_like(inputs) * noise_factor
            noisy = torch.clip(noisy,0.,1.)
            return noisy
            
        noise_imgs = [add_noise(T.ToTensor()(orig_img),noise_factor) for noise_factor in (0.3,0.6,0.9)]
        return [T.ToPILImage()(noise_img) for noise_img in noise_imgs]
# Random Blocks
    def resize(self, orig_img):
        def add_random_boxes(img,n_k,size=32):
            h,w = size,size
            img = np.asarray(img)
            img_size = img.shape[1]
            boxes = []
            for k in range(n_k):
                y,x = np.random.randint(0,img_size-w,(2,))
                img[y:y+h,x:x+w] = 0
                boxes.append((x,y,h,w))
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            return img
        return [add_random_boxes(orig_img,n_k=i) for i in (10,20)]
# Central Region
    def resize(self, orig_img):
        def add_central_region(img,size=32):
            h,w = size,size
            img = np.asarray(img)
            img_size = img.shape[1] 
            img[int(img_size/2-h):int(img_size/2+h),int(img_size/2-w):int(img_size/2+w)] = 0
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            return img
        return [add_central_region(orig_img,size=s) for s in (32,64)]