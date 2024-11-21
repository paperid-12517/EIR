import numpy as np
import kornia
import torch
import random
import math
import torch.nn as nn
from scipy import ndimage
from utils import mixstyle
import torch.nn.functional as F


def colorJitter(colorJitter, img_mean, data = None, target = None, s=0.25):
    # s is the strength of colorjitter1
    #colorJitter
    if not (data is None):
        if data.shape[1]==3:
            if colorJitter > 0.2:
                img_mean, _ = torch.broadcast_tensors(img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3), data)
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s,hue=s))
                data = (data+img_mean)/255
                data = seq(data)
                data = (data*255-img_mean).float()
    return data, target

def gaussian_blur(blur, data = None, target = None):
    if not (data is None):
        if data.shape[1]==3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15,1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

def flip(flip, data = None, target = None):
    #Flip
    if flip == 1:
        if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
        if not (target is None):
            target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def p_Mix_up(mask, data = None,target = None, num1 = 21):
    # Mix
    alpha = 1.0
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    if not (data is None):
      
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
     
        data = (lam * stackedMask0 * data[0] + (1-lam) * stackedMask0 * data[1] + (1 - stackedMask0) * data[1]).unsqueeze(0)      
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (lam * stackedMask0 * target[0]  +  (1-lam) * stackedMask0 * target[1] +  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target

def Mix_up_bg(mask, data = None,target = None, num1 = 21):
    alpha = 1.0
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (lam * stackedMask0 * data[0] + (1-lam) * stackedMask0 * data[1] + (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])

        target = (lam * stackedMask0 * target[0]  +  (1-lam) * stackedMask0 * target[1] +  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target

def instance_replay(mask, data = None):
    if not (data is None):    
        stackedMask0, _ = torch.broadcast_tensors(mask, data)   
        data = (stackedMask0*data)
    return data

def p_Mix_o(mask,data = None, target = None, position = None,mask_n = None,width=None,height=None):
    new_target_x = position[0]
    new_target_y = position[1]
    if not (data is None): 
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0]) #(3,513,513)

        mask_copy = stackedMask0.clone()
        mask_np = mask_copy.cpu().numpy()
        labeled_image, num_features = ndimage.label(mask_np == 1) 

        largest_box = find_largest_bounding_box(mask[0])
        min_x, min_y, max_x, max_y = largest_box[0],largest_box[1],largest_box[2],largest_box[3] 
        scale_factor = 1.1
        scale_factor_w = width / (max_x - min_x) if (max_x - min_x) != 0 else 1.1
        scale_factor_h = height / (max_y - min_y) if (max_y - min_y) != 0 else 1.1

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2 
        new_min_x, new_min_y = int(center_x - (center_x - min_x) * scale_factor_w), int(center_y - (center_y - min_y) * scale_factor_h)
        new_max_x, new_max_y = int(center_x + (max_x - center_x) * scale_factor_w), int(center_y + (max_y - center_y) * scale_factor_h)
        current_centroid_x = (new_min_x + new_max_x) / 2
        current_centroid_y = (new_min_y + new_max_y) / 2

        offset_x = new_target_x - current_centroid_x
        offset_y = new_target_y - current_centroid_y
        new_min_xf = new_min_x + offset_x
        new_min_yf = new_min_y + offset_y
        new_max_xf = new_max_x + offset_x
        new_max_yf = new_max_y + offset_y
        new_min_yf = int(round(new_min_yf))
        new_max_yf = int(round(new_max_yf))
        new_min_xf = int(round(new_min_xf))
        new_max_xf = int(round(new_max_xf))

        region_to_zoom = mask_np[:, min_y:max_y + 1, min_x:max_x + 1]
        resized_region_np = ndimage.zoom(region_to_zoom, (1, scale_factor_h, scale_factor_w),order=0)

        if(resized_region_np.shape[1]>512 or resized_region_np.shape[2]>512):
            resized_region_np = mask_np[:, min_y:max_y + 1, min_x:max_x + 1]

        if((new_min_yf + resized_region_np.shape[1])>512):
            new_min_yf = 511-resized_region_np.shape[1]
        if((new_min_xf + resized_region_np.shape[2]) > 512):
            new_min_xf = 511-resized_region_np.shape[2]
        if (new_min_yf < 0):
            new_min_yf = 0
        if (new_min_xf < 0):
            new_min_xf = 0

        mask_np.fill(0)
        mask_np[:, new_min_yf:new_min_yf + resized_region_np.shape[1], new_min_xf:new_min_xf + resized_region_np.shape[2]] = resized_region_np


        mask_final = torch.from_numpy(mask_np)
        data_np = data[0].cpu().numpy()
        data_to_zoom = data_np[:,min_y:max_y + 1, min_x:max_x + 1]
        data_np_1 = ndimage.zoom(data_to_zoom, (1, scale_factor_h, scale_factor_w), order=0)

        if (data_np_1.shape[1] > 512 or data_np_1.shape[2] > 512):
            data_np_1 = data_np[:, min_y:max_y + 1, min_x:max_x + 1]
        data_np.fill(0)
        data_np[:, new_min_yf:new_min_yf + resized_region_np.shape[1], new_min_xf:new_min_xf + resized_region_np.shape[2]] = data_np_1

        data0 = torch.from_numpy(data_np).cuda()

        MixMask_nn = 1 - mask_n
        MixMask_nn_0, _ = torch.broadcast_tensors(MixMask_nn, data[0])  #(3,513,513)
        mask_final = mask_final.cuda()
        Mask_final = (mask_final & MixMask_nn_0).float().int().cuda()
        data = data0*Mask_final + (1-Mask_final)*data[1]

    if not (target is None): 
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0]) # (513,513)
        mask_copy = stackedMask0.clone() 
        target_r = stackedMask0 * target[0]  # (513,513)
        target_region = torch.where(target_r != 0)
        mask_np = mask_copy.cpu().numpy()  
        labeled_image, num_features = ndimage.label(mask_np == 1) 

        largest_box = find_largest_bounding_box(mask[0])  # (min_x, min_y, max_x, max_y)
        min_x, min_y, max_x, max_y = largest_box[0], largest_box[1], largest_box[2], largest_box[3]  
        scale_factor = 1.1
        scale_factor_w = width / (max_x - min_x) if (max_x - min_x) != 0 else 1.1
        scale_factor_h = height / (max_y - min_y) if (max_y - min_y) != 0 else 1.1
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        new_min_x, new_min_y = int(center_x - (center_x - min_x) * scale_factor_w), int(center_y - (center_y - min_y) * scale_factor_h)
        new_max_x, new_max_y = int(center_x + (max_x - center_x) * scale_factor_w), int(center_y + (max_y - center_y) * scale_factor_h)

        current_centroid_x = (new_min_x + new_max_x) / 2
        current_centroid_y = (new_min_y + new_max_y) / 2

        offset_x = new_target_x - current_centroid_x
        offset_y = new_target_y - current_centroid_y
        region_to_zoom = mask_np[min_y:max_y + 1, min_x:max_x + 1]  
        resized_region_np = ndimage.zoom(region_to_zoom, (scale_factor_h, scale_factor_w),order=0)  

        new_min_xf = new_min_x + offset_x
        new_min_yf = new_min_y + offset_y

        new_max_xf = new_max_x + offset_x
        new_max_yf = new_max_y + offset_y

        new_min_yf = int(round(new_min_yf))
        new_max_yf = int(round(new_max_yf))
        new_min_xf = int(round(new_min_xf))
        new_max_xf = int(round(new_max_xf))

        if (resized_region_np.shape[0] > 512 or resized_region_np.shape[1] > 512):
            resized_region_np = mask_np[min_y:max_y + 1, min_x:max_x + 1]

        if ((new_min_yf + resized_region_np.shape[0]) > 512):
            new_min_yf = 511 - resized_region_np.shape[0]
        if ((new_min_xf + resized_region_np.shape[1]) > 512):
            new_min_xf = 511 - resized_region_np.shape[1]
        if (new_min_yf< 0):
            new_min_yf = 0
        if (new_min_xf< 0):
            new_min_xf = 0

        mask_np.fill(0)
        mask_np[new_min_yf:new_min_yf + resized_region_np.shape[0], new_min_xf:new_min_xf + resized_region_np.shape[1]] = resized_region_np
        mask_final = torch.from_numpy(mask_np)

        target_np = target[0].cpu().numpy()
        nonzero_count_200 = np.count_nonzero(target_np)
        target_np_zoom = target_np[min_y:max_y + 1, min_x:max_x + 1]
        target_np_1 = ndimage.zoom(target_np_zoom,(scale_factor_h, scale_factor_w), order=0)
        if (target_np_1.shape[0] > 512 or target_np_1.shape[1] > 512):
            target_np_1 = target_np[min_y:max_y + 1, min_x:max_x + 1]

        target_np.fill(0)
        target_np[new_min_yf:new_min_yf + resized_region_np.shape[0], new_min_xf:new_min_xf + resized_region_np.shape[1]] = target_np_1

        target0 = torch.from_numpy(target_np).cuda()

        MixMask_nn = 1 - mask_n
        MixMask_nn_0,_ = torch.broadcast_tensors(MixMask_nn, target[0])
        mask_final = mask_final.cuda()

        Mask_final = (mask_final & MixMask_nn_0).float().int().cuda()
        target = target0 * Mask_final + (1 - Mask_final) * target[1]

    return data, target

def p_Mix_bg(mask,data = None, target = None, position = None,mask_n = None):

    if not (data is None): 
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1]).unsqueeze(0)

    if not (target is None):  
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(0)

    return data, target

def find_largest_bounding_box(mask_np = None):

    largest_box = None
    largest_box_size = 0
    labeled_image_n, num_features_n = ndimage.label(mask_np == 1) 

    #bianli every range to find the largest_box
    for i in range(1, num_features_n + 1):
        # ys, xs = np.where(labeled_image == i)
        ys,xs = np.where(labeled_image_n == i)
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        box_size = (max_x - min_x + 1) * (max_y - min_y + 1)
        if box_size > largest_box_size:
            largest_box_size = box_size
            largest_box = (min_x, min_y, max_x, max_y)
    return largest_box

def compute_centroids(label_image, target_class):
  
    target_positions = torch.where((label_image == target_class))
    centroid_x = torch.mean(target_positions[1].float())
    centroid_y = torch.mean(target_positions[0].float())

    centroid = (centroid_x.item(), centroid_y.item())
    return centroid

def normalize(MEAN, STD, data = None, target = None):
    #Normalize
    if not (data is None):
        if data.shape[1]==3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data-MEAN)/STD).float()
    return data, target

def masaike(mask, data = None, target = None, position = None, mask_n = None, width=256, height=256):

    if position == 0:
        new_target_x = 128
        new_target_y = 128
    elif position == 1:
        new_target_x = 384
        new_target_y = 128
    elif position == 2:
        new_target_x = 128
        new_target_y = 384
    elif position == 3:
        new_target_x = 384
        new_target_y = 384

    if not (data is None):   
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0]) #(3,513,513)        
        mask_copy = stackedMask0.clone()        
        mask_np = mask_copy.cpu().numpy()       
        labeled_image, num_features = ndimage.label(mask_np[0] == 1)                      
        largest_box = find_largest_bounding_box(mask_np[0]) #(min_x, min_y, max_x, max_y)  

        if largest_box is not None:
            min_x, min_y, max_x, max_y = largest_box[0],largest_box[1],largest_box[2],largest_box[3]  
        else:
            count_ones = np.sum(mask_np == 1)
            print(count_ones)
        scale_factor = 1.1
        scale_factor_w = width/(max_x - min_x) if (max_x - min_x) != 0 else scale_factor
        scale_factor_h = height/(max_y - min_y) if (max_y - min_y) != 0 else scale_factor
        scale_factor_final = min(scale_factor_w, scale_factor_h)            

        region_to_zoom = mask_np[:, min_y:max_y + 1, min_x:max_x + 1]       
        resized_region_np = ndimage.zoom(region_to_zoom, (1, scale_factor_final, scale_factor_final),order=0)   

        if(resized_region_np.shape[1]>512 or resized_region_np.shape[2]>512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])  
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w)
            resized_region_np = ndimage.zoom(mask_np[:, min_y:max_y + 1, min_x:max_x + 1], (1, scale_factor_final, scale_factor_final), order=0)

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2      
        new_min_x, new_min_y = int(center_x - (center_x - min_x) * scale_factor_final), int(center_y - (center_y - min_y) * scale_factor_final)
        new_max_x, new_max_y = int(center_x + (max_x - center_x) * scale_factor_final), int(center_y + (max_y - center_y) * scale_factor_final)
    
        current_centroid_x = (new_min_x + new_max_x) / 2
        current_centroid_y = (new_min_y + new_max_y) / 2
        offset_x = new_target_x - current_centroid_x
        offset_y = new_target_y - current_centroid_y

        new_min_xf = new_min_x + offset_x
        new_min_yf = new_min_y + offset_y
        new_max_xf = new_max_x + offset_x
        new_max_yf = new_max_y + offset_y
        new_min_yf = int(round(new_min_yf))
        new_max_yf = int(round(new_max_yf))
        new_min_xf = int(round(new_min_xf))
        new_max_xf = int(round(new_max_xf))
        
        if((new_min_yf + resized_region_np.shape[1])>512):
            new_min_yf = 511-resized_region_np.shape[1]
        if((new_min_xf + resized_region_np.shape[2]) > 512):
            new_min_xf = 511-resized_region_np.shape[2]
        if (new_min_yf < 0):
            new_min_yf = 0
        if (new_min_xf < 0):
            new_min_xf = 0

        mask_np.fill(0)
        mask_np[:, new_min_yf:new_min_yf + resized_region_np.shape[1], new_min_xf:new_min_xf + resized_region_np.shape[2]] = resized_region_np

        mask_final = torch.from_numpy(mask_np)

        data_np = data[0].cpu().numpy()
        data_to_zoom = data_np[:,min_y:max_y + 1, min_x:max_x + 1]
        data_np_1 = ndimage.zoom(data_to_zoom, (1, scale_factor_final, scale_factor_final), order=0)

        if (data_np_1.shape[1] > 512 or data_np_1.shape[2] > 512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w) 
            data_np_1 = ndimage.zoom(data_np[:,min_y:max_y + 1, min_x:max_x + 1], (1, scale_factor_final, scale_factor_final), order=0)
        data_np.fill(0)
        data_np[:, new_min_yf:new_min_yf + resized_region_np.shape[1], new_min_xf:new_min_xf + resized_region_np.shape[2]] = data_np_1      
        data0 = torch.from_numpy(data_np).cuda()

        MixMask_nn = 1 - mask_n
        MixMask_nn_0, _ = torch.broadcast_tensors(MixMask_nn, data[0])  #(3,513,513)
        mask_final = mask_final.cuda()
        Mask_final = (mask_final & MixMask_nn_0).float().int().cuda()
        data = data0*Mask_final + (1-Mask_final)*data[1]

    if not (target is None): 
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0]) # (513,513)
        mask_copy = stackedMask0.clone() # (513,513)
        mask_np = mask_copy.cpu().numpy()  
        target_r = stackedMask0 * target[0]  # (513,513)
        target_region = torch.where(target_r != 0)
        labeled_image, num_features = ndimage.label(mask_np == 1)  
        largest_box = find_largest_bounding_box(mask_np) 

        min_x, min_y, max_x, max_y = largest_box[0], largest_box[1], largest_box[2], largest_box[3]  
        scale_factor = 1.1
        scale_factor_w = width / (max_x - min_x) if (max_x - min_x) != 0 else scale_factor
        scale_factor_h = height / (max_y - min_y) if (max_y - min_y) != 0 else scale_factor
        scale_factor_final = min(scale_factor_w, scale_factor_h) 

        region_to_zoom = mask_np[min_y:max_y + 1, min_x:max_x + 1] 
        resized_region_np = ndimage.zoom(region_to_zoom, (scale_factor_final, scale_factor_final),order=0) 
        if (resized_region_np.shape[0] > 512 or resized_region_np.shape[1] > 512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w)
            resized_region_np = ndimage.zoom(region_to_zoom, (scale_factor_final, scale_factor_final),order=0)

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        new_min_x, new_min_y = int(center_x - (center_x - min_x) * scale_factor_final), int(center_y - (center_y - min_y) * scale_factor_final)
        new_max_x, new_max_y = int(center_x + (max_x - center_x) * scale_factor_final), int(center_y + (max_y - center_y) * scale_factor_final)
        current_centroid_x = (new_min_x + new_max_x) / 2
        current_centroid_y = (new_min_y + new_max_y) / 2
        offset_x = new_target_x - current_centroid_x
        offset_y = new_target_y - current_centroid_y

        new_min_xf = new_min_x + offset_x
        new_min_yf = new_min_y + offset_y

        new_max_xf = new_max_x + offset_x
        new_max_yf = new_max_y + offset_y

        new_min_yf = int(round(new_min_yf))
        new_max_yf = int(round(new_max_yf))
        new_min_xf = int(round(new_min_xf))
        new_max_xf = int(round(new_max_xf))

        if ((new_min_yf + resized_region_np.shape[0]) > 512):
            new_min_yf = 511 - resized_region_np.shape[0]
        if ((new_min_xf + resized_region_np.shape[1]) > 512):
            new_min_xf = 511 - resized_region_np.shape[1]
        if (new_min_yf< 0):
            new_min_yf = 0
        if (new_min_xf< 0):
            new_min_xf = 0

        mask_np.fill(0)
        mask_np[new_min_yf:new_min_yf + resized_region_np.shape[0], new_min_xf:new_min_xf + resized_region_np.shape[1]] = resized_region_np
   
        mask_final = torch.from_numpy(mask_np)

        target_np = target[0].cpu().numpy()
        nonzero_count_200 = np.count_nonzero(target_np)
        # print(nonzero_count_200)
        target_np_zoom = target_np[min_y:max_y + 1, min_x:max_x + 1]
        target_np_1 = ndimage.zoom(target_np_zoom,(scale_factor_final, scale_factor_final), order=0)
      
        if (target_np_1.shape[0] > 512 or target_np_1.shape[1] > 512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w)
            target_np_1 = ndimage.zoom(target_np_zoom,(scale_factor_final, scale_factor_final), order=0)

        target_np.fill(0)
        target_np[new_min_yf:new_min_yf + resized_region_np.shape[0], new_min_xf:new_min_xf + resized_region_np.shape[1]] = target_np_1

        target0 = torch.from_numpy(target_np).cuda()

        MixMask_nn = 1 - mask_n
        MixMask_nn_0,_ = torch.broadcast_tensors(MixMask_nn, target[0]) 
        mask_final = mask_final.cuda()

        Mask_final = (mask_final & MixMask_nn_0).float().int().cuda()
        target = target0 * Mask_final + (1 - Mask_final) * target[1]

    return data, target

def masaike1(mask, data = None, target = None, position = None, mask_n = None, width=256, height=256):

    if position == 0:
        new_target_x = 128
        new_target_y = 128
    elif position == 1:
        new_target_x = 384
        new_target_y = 128
    elif position == 2:
        new_target_x = 128
        new_target_y = 384
    elif position == 3:
        new_target_x = 384
        new_target_y = 384

    if not (data is None):    

        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0]) #(3,513,513)       
        mask_copy = stackedMask0.clone()       
        mask_np = mask_copy.cpu().numpy()        
        labeled_image, num_features = ndimage.label(mask_np[0] == 1)                      
        largest_box = find_largest_bounding_box(mask_np[0]) 

        if largest_box is not None:
            min_x, min_y, max_x, max_y = largest_box[0],largest_box[1],largest_box[2],largest_box[3]  
        else:
            count_ones = np.sum(mask_np == 1)
            print(count_ones)
        scale_factor = 1.1
        scale_factor_w = width/(max_x - min_x) if (max_x - min_x) != 0 else scale_factor
        scale_factor_h = height/(max_y - min_y) if (max_y - min_y) != 0 else scale_factor
        scale_factor_final = min(scale_factor_w, scale_factor_h)           

        region_to_zoom = mask_np[:, min_y:max_y + 1, min_x:max_x + 1]       
        resized_region_np = ndimage.zoom(region_to_zoom, (1, scale_factor_final, scale_factor_final),order=0)  

        if(resized_region_np.shape[1]>512 or resized_region_np.shape[2]>512): 
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])  
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w)
            resized_region_np = ndimage.zoom(mask_np[:, min_y:max_y + 1, min_x:max_x + 1], (1, scale_factor_final, scale_factor_final), order=0)

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2      
        new_min_x, new_min_y = int(center_x - (center_x - min_x) * scale_factor_final), int(center_y - (center_y - min_y) * scale_factor_final)
        new_max_x, new_max_y = int(center_x + (max_x - center_x) * scale_factor_final), int(center_y + (max_y - center_y) * scale_factor_final)

        current_centroid_x = (new_min_x + new_max_x) / 2
        current_centroid_y = (new_min_y + new_max_y) / 2

        offset_x = new_target_x - current_centroid_x
        offset_y = new_target_y - current_centroid_y

        new_min_xf = new_min_x + offset_x
        new_min_yf = new_min_y + offset_y
        new_max_xf = new_max_x + offset_x
        new_max_yf = new_max_y + offset_y
        new_min_yf = int(round(new_min_yf))
        new_max_yf = int(round(new_max_yf))
        new_min_xf = int(round(new_min_xf))
        new_max_xf = int(round(new_max_xf))

        if((new_min_yf + resized_region_np.shape[1])>512):
            new_min_yf = 511-resized_region_np.shape[1]
        if((new_min_xf + resized_region_np.shape[2]) > 512):
            new_min_xf = 511-resized_region_np.shape[2]
        if (new_min_yf < 0):
            new_min_yf = 0
        if (new_min_xf < 0):
            new_min_xf = 0

        mask_np.fill(0)
        mask_np[:, new_min_yf:new_min_yf + resized_region_np.shape[1], new_min_xf:new_min_xf + resized_region_np.shape[2]] = 1     

        data_np = data[0].cpu().numpy()
        data_to_zoom = data_np[:,min_y:max_y + 1, min_x:max_x + 1]
        data_np_1 = ndimage.zoom(data_to_zoom, (1, scale_factor_final, scale_factor_final), order=0)

        if (data_np_1.shape[1] > 512 or data_np_1.shape[2] > 512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w) 
            data_np_1 = ndimage.zoom(data_np[:,min_y:max_y + 1, min_x:max_x + 1], (1, scale_factor_final, scale_factor_final), order=0)

        data2_np = data[1].cpu().numpy()
        data2_np[:, new_min_yf:new_min_yf + resized_region_np.shape[1], new_min_xf:new_min_xf + resized_region_np.shape[2]] = data_np_1

        data = torch.from_numpy(data2_np).cuda()

    if not (target is None):  
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0]) # (513,513)
        mask_copy = stackedMask0.clone() # (513,513)
        mask_np = mask_copy.cpu().numpy() 
        target_r = stackedMask0 * target[0]  # (513,513)
        target_region = torch.where(target_r != 0)
        labeled_image, num_features = ndimage.label(mask_np == 1)  
        largest_box = find_largest_bounding_box(mask_np) 

        min_x, min_y, max_x, max_y = largest_box[0], largest_box[1], largest_box[2], largest_box[3] 
        scale_factor = 1.1
        scale_factor_w = width / (max_x - min_x) if (max_x - min_x) != 0 else scale_factor
        scale_factor_h = height / (max_y - min_y) if (max_y - min_y) != 0 else scale_factor
        scale_factor_final = min(scale_factor_w, scale_factor_h) 

        region_to_zoom = mask_np[min_y:max_y + 1, min_x:max_x + 1]  
        resized_region_np = ndimage.zoom(region_to_zoom, (scale_factor_final, scale_factor_final),order=0) 

        if (resized_region_np.shape[0] > 512 or resized_region_np.shape[1] > 512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w)
            resized_region_np = ndimage.zoom(region_to_zoom, (scale_factor_final, scale_factor_final),order=0)

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        new_min_x, new_min_y = int(center_x - (center_x - min_x) * scale_factor_final), int(center_y - (center_y - min_y) * scale_factor_final)
        new_max_x, new_max_y = int(center_x + (max_x - center_x) * scale_factor_final), int(center_y + (max_y - center_y) * scale_factor_final)

        current_centroid_x = (new_min_x + new_max_x) / 2
        current_centroid_y = (new_min_y + new_max_y) / 2

        offset_x = new_target_x - current_centroid_x
        offset_y = new_target_y - current_centroid_y

        new_min_xf = new_min_x + offset_x
        new_min_yf = new_min_y + offset_y

        new_max_xf = new_max_x + offset_x
        new_max_yf = new_max_y + offset_y

        new_min_yf = int(round(new_min_yf))
        new_max_yf = int(round(new_max_yf))
        new_min_xf = int(round(new_min_xf))
        new_max_xf = int(round(new_max_xf))

        if ((new_min_yf + resized_region_np.shape[0]) > 512):
            new_min_yf = 511 - resized_region_np.shape[0]
        if ((new_min_xf + resized_region_np.shape[1]) > 512):
            new_min_xf = 511 - resized_region_np.shape[1]
        if (new_min_yf< 0):
            new_min_yf = 0
        if (new_min_xf< 0):
            new_min_xf = 0

        mask_np.fill(0)
        mask_np[new_min_yf:new_min_yf + resized_region_np.shape[0], new_min_xf:new_min_xf + resized_region_np.shape[1]] = resized_region_np

        target_np = target[0].cpu().numpy()
        nonzero_count_200 = np.count_nonzero(target_np)
        target_np_zoom = target_np[min_y:max_y + 1, min_x:max_x + 1]
        target_np_1 = ndimage.zoom(target_np_zoom,(scale_factor_final, scale_factor_final), order=0)
      
        if (target_np_1.shape[0] > 512 or target_np_1.shape[1] > 512):
            new_scale_factor_h = min(scale_factor_final, 512 / resized_region_np.shape[1])  
            new_scale_factor_w = min(scale_factor_final, 512 / resized_region_np.shape[2])
            scale_factor_final = min(new_scale_factor_h, new_scale_factor_w)
            target_np_1 = ndimage.zoom(target_np_zoom,(scale_factor_final, scale_factor_final), order=0)

        target2_np = target[1].cpu().numpy()
        target2_np[new_min_yf:new_min_yf + resized_region_np.shape[0], new_min_xf:new_min_xf + resized_region_np.shape[1]] = target_np_1
        target = torch.from_numpy(target2_np).cuda()

    return data, target


def p_Mixup_beijing(mask, data = None, target_0 = None, target_1 = None,mask_nn = None,found_region = None, num1 = 21):
    alpha = 1.0
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    found_region = list(found_region)
    target_size = (250, 250)  

    mask_copy = mask[0].clone()      #[512,512]
    indices = mask_copy.nonzero()
    if indices.numel() == 0:        
        bbox = torch.tensor([-1,-1,-1,-1])
        return data[1], target_1
    else:
        min_y,min_x = indices.min(dim = 0).values   
        max_y,max_x = indices.max(dim = 0).values
        bbox = torch.tensor([min_x.item(), min_y.item(), max_x.item() + 1, max_y.item() + 1])    
    max_x, min_x, max_y, min_y = max_x.item(), min_x.item(), max_y.item(), min_y.item()
    scale_factor = 1
    scale_factor_w = target_size[0]/(max_x - min_x + 2) if (max_x - min_x) != 0 else scale_factor
    scale_factor_h = target_size[1]/(max_y - min_y + 2) if (max_y - min_y) != 0 else scale_factor

    if (max_x - min_x) == 0 or (max_y - min_y) == 0:
        return data[1], target_1
    scale_factor_final = min(scale_factor_w, scale_factor_h)         

    if not (data is None):   

        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])    #[3,512,512]
        region_to_zoom = stackedMask0[:, min_y:max_y + 1, min_x:max_x + 1]     #[3,106,258]    
        region_to_zoom_4d = region_to_zoom.unsqueeze(0)                        #[1,3,106,258]
        resized_region = F.interpolate(region_to_zoom_4d.float(), scale_factor=scale_factor_final, mode='nearest').long()       
        resized_region = resized_region.squeeze(0)      #[3,81,199]
       
        mask_final = torch.zeros_like(stackedMask0, dtype=stackedMask0.dtype)   #[3,512,512]
        if (found_region[0]+resized_region.shape[1])>512:
            found_region[0] = 511-resized_region.shape[1]
        if (found_region[1]+resized_region.shape[2])>512:
            found_region[1] = 511-resized_region.shape[2]
        mask_final[:, found_region[0]:found_region[0]+resized_region.shape[1], found_region[1]:found_region[1]+resized_region.shape[2]] = resized_region  #[3,512,512]

        data_to_zoom = data[0][:,min_y:max_y + 1, min_x:max_x + 1]          #[3,106,258]
        data_to_zoom_4d = data_to_zoom.unsqueeze(0)                         #[1,3,106,258]
        data_resized_region = F.interpolate(data_to_zoom_4d, scale_factor=scale_factor_final, mode='nearest')     #[1,3,81,199]
        data_resized_region = data_resized_region.squeeze(0)                #[3,81,199]

        data0[:, found_region[0]:found_region[0]+data_resized_region.shape[1], found_region[1]:found_region[1]+data_resized_region.shape[2]] = data_resized_region    
        Mask_final = mask_final.cuda()
        data = (lam*Mask_final*data0 + (1-lam)*Mask_final*data[1]) + (1-Mask_final)*data[1]

    if not (target_0 is None): 
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target_0)           # [512,512]
        region_to_zoom1 = stackedMask0[min_y:max_y + 1, min_x:max_x + 1]       # [106,258] 
        region_to_zoom1_4d = region_to_zoom1.unsqueeze(0).unsqueeze(0)
        resized_region1 = F.interpolate(region_to_zoom1_4d.float(),scale_factor=scale_factor_final, mode= 'nearest').long()     
        resized_region1 = resized_region1.squeeze(0).squeeze(0)  

        if (found_region[0]+resized_region1.shape[0])>512:
            found_region[0] = 511-resized_region1.shape[0]
        if (found_region[1]+resized_region1.shape[1])>512:
            found_region[1] = 511-resized_region1.shape[1]

        mask_final = torch.zeros_like(stackedMask0, dtype=stackedMask0.dtype)   # [512,512]
        mask_final[found_region[0]:found_region[0]+resized_region1.shape[0], found_region[1]:found_region[1]+resized_region1.shape[1]] = resized_region1

        target_to_zoom = target_0[min_y:max_y + 1, min_x:max_x + 1]         #[106,258]
        target_to_zoom_4d = target_to_zoom.unsqueeze(0).unsqueeze(0) #[1,1,513,513]
        target_resized_region = F.interpolate(target_to_zoom_4d.float(),scale_factor=scale_factor_final, mode= 'nearest').long()    
        target_resized_region = target_resized_region.squeeze(0).squeeze(0)

        target0 = torch.zeros_like(target_0, dtype=target_0.dtype)
        target0[found_region[0]:found_region[0]+target_resized_region.shape[0], found_region[1]:found_region[1]+target_resized_region.shape[1]] = target_resized_region

        Mask_final = mask_final.cuda()    

        target0 = target0.cuda()
        target_1 = target_1.cuda()
        n_cl = torch.tensor(num1).cuda()
        m_label_copy = target0.clone().to('cuda', non_blocking=True)  # (513,513)
        labels_new_2 = torch.where(m_label_copy != 255, m_label_copy, n_cl)  
        m_label_copy = F.one_hot(labels_new_2, num1 + 1).float().permute(2, 0, 1)   
        m_label_copy = m_label_copy[:num1, :, :] 
        M_final = Mask_final.unsqueeze(0).repeat(num1,1,1)          # [21,512,512]
        target =(lam * M_final * m_label_copy ) + ((1-lam) * M_final * target_1) + (1 - M_final) * target_1      

    return data, target