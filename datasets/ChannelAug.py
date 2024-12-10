from __future__ import absolute_import

from torchvision.transforms import *

#from PIL import Image
import random
import math
import torch
import torch.nn as nn
#import numpy as np
#import torch

class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val = 'inside_random', spacing='random', transforms_like=False, p_random_apply_rand_quant = 1):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0] # min, max over batch size, spatial dimension
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) # -> (C), (C), (C)

        ##region percentiles for each channel
        if self.spacing == "random":
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            region_percentiles = torch.tile(torch.arange(1/(total_region_percentile_number_per_channel[0] + 1), 1, step=1/(total_region_percentile_number_per_channel[0]+1), device=x.device), [C])
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        # ordered region ends
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # ordered middle points
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        # associate region id
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            # middle points as the proxy for all values in corresponding regions
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(), device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (ordered_region_right_ends - ordered_region_left_ends)
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = torch.where(torch.rand([B,1,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = torch.where(torch.rand([C,1,1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)

        return x

class ChannelAdap(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            img = img

        return img
        
        
class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 3)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
                img[0,:,:] = tmp_img
                img[1,:,:] = tmp_img
                img[2,:,:] = tmp_img
        return img

class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img
    
class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img