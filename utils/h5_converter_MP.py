import h5py
import torch
from monai.transforms import (
    SpatialCropd,
    Compose,
    RandShiftIntensityd
)

import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Resized, Compose, LoadImaged, Spacingd, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, CropForegroundd, SpatialCropd, CenterSpatialCropd, SpatialPadd
import nibabel as nib



# def load_file(input_dict):
    
#     data_dict = {
#         'image': torch.tensor(nib.load(input_dict['image']).get_fdata()), 
#         'label': convert_label(torch.tensor(nib.load(input_dict['label']).get_fdata()))
#     }
#     return data_dict



class Load_File(object):
    """load the file dir dict and convert to actualy file dir"""
    
    def load_file(self, input_dict):
    
        data_dict = {
            'image': torch.tensor(nib.load(input_dict['image']).get_fdata()).unsqueeze(0), 
            'label': convert_label(torch.tensor(nib.load(input_dict['label']).get_fdata())).unsqueeze(0)
        }
        return data_dict

    def __call__(self, input_dict):
        return self.load_file(input_dict)


data_reader = Compose(
    [
        Load_File(),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(96, 96, 64)),
        
        CenterSpatialCropd(keys=["image", "label"], roi_size=(64, 64, 64)),
    ]
)



def convert_label(input_label):
    input_label_5 = (input_label == 3).float()
    return input_label_5


def convert_h5(dir, des_dir):
    for img_name in os.listdir(dir):
        # make sure all data are CT data
        if img_name.endswith('img.nii'):
            image_index = img_name[:6]
            img_dir = os.path.join(dir, image_index + '_img.nii')
            label_dir = os.path.join(dir, image_index + '_mask.nii')

            dir_dict = {
                'image' : img_dir,
                'label' : label_dir
            }
            
            print(dir_dict)
            
            loaded_dict = data_reader(dir_dict)
            print(loaded_dict['image'].shape)
            with h5py.File(os.path.join(des_dir, image_index + '.h5'), 'w') as hf:
                hf.create_dataset('image', data=loaded_dict['image'])
                hf.create_dataset('label', data=(loaded_dict['label'] > 0.5).float())
                
convert_h5('data/data_MP', 'data/MP_H5')