import torch
import nibabel as nib
from monai import transforms
from glob import glob
import torch
import os
import h5py

class ConverToBinaryLabeld(transforms.Transform):
    
    def __init__(self, source_key="label"):
        """transform that will convert multi-label to binary label

        Args:
            source_key (str, optional): key of the dictionary. Defaults to "label".
        """
        super().__init__()
        self.source_key = source_key
        
    
    
    def __call__(self, input_dict):
        label = input_dict[self.source_key]
        print(torch.unique(label))
        label = (label != 0).to(label.dtype)
        label = label.unsqueeze(0)
        input_dict[self.source_key] = label
        
        return input_dict
    


def get_train_transform():
      
    transform = transforms.Compose([
            transforms.LoadImaged(keys = ["image", "label"], reader="NibabelReader"),
            ConverToBinaryLabeld(),
            transforms.Resized(keys = ["image", "label"], spatial_size=(80, 80, 80)),
            transforms.CenterSpatialCropd(keys = ["image", "label"], roi_size=(64, 64, 64))
        ])
    return transform
    

# return a list of dictionary
def get_data_list_BraTs(root_dir):
    data_dirs = os.listdir(root_dir)
    data_dirs = [data_dir for data_dir in data_dirs if data_dir.startswith("BraTS")]
    data_dict_dir = []
    for data_dir in data_dirs:
        index = data_dir
        data_dir = os.path.join(root_dir, data_dir)
    
    
    
        label_path = glob(os.path.join(data_dir, '*seg*'))
        img_path = [file_dir for file_dir in glob(os.path.join(data_dir, '*')) if file_dir not in label_path]
        data_dict_dir.append({
            'index': index,
            'label': label_path,
            'image': img_path
        })
    
    return data_dict_dir



def convert_h5(root_dir: str, des_dir: str, data_reader: transforms, test_function=True):
    """convert BraTs dataset into h5 format with index as name (00032.h5)

    Args:
        root_dir (str): _description_
        des_dir (str): _description_
        data_reader (transforms): _description_
    """
    
    data_dict_list = get_data_list_BraTs(root_dir)
    if test_function:
        data_dict_list = data_dict_list[:5]
    print('Processing ...')
    for data in data_dict_list:
        loaded_dict = data_reader(data)
        with h5py.File(os.path.join(des_dir, 'BarTs' + loaded_dict['index'] + '.h5'), 'w') as hf:
            hf.create_dataset('image', data=loaded_dict['image'])
            hf.create_dataset('label', data=(loaded_dict['label']))
    
    print('Done')


if __name__ == "__main__":
    # data_dict_list = get_data_list_BraTs(r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData')
    import matplotlib.pyplot as plt
    convert_h5(
        '/home/xiangcen/SPRV_Brain/data/ISBI2024-BraTS-GoAT-TrainingData',
        'data/BraTs_H5',
        get_train_transform(),
        False
    )
    
