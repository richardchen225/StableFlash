# A reimplemented version in public environments by Xiao Fu and Mu Hu

import torch.nn.functional as F
import sys
sys.path.append("..")

def resize_max_res_tensor(input_tensor, mode, recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    downscale_factor = min(recom_resolution/original_H, recom_resolution/original_W)
    
    if mode == 'normal':
        resized_input_tensor = F.interpolate(input_tensor,
                                            scale_factor=downscale_factor,
                                            mode='nearest')
    else:
        resized_input_tensor = F.interpolate(input_tensor,
                                            scale_factor=downscale_factor,
                                            mode='bilinear',
                                            align_corners=False)

    if mode == 'depth':
        return resized_input_tensor / downscale_factor
    else:
        return resized_input_tensor
