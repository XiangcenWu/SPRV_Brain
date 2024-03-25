import torch
import h5py
from monai.transforms import Transform



class ReadH5d(Transform):
    """Convert h5 file to dict of Tensors."""
    def __init__(self):
        super().__init__()

    def __call__(self, file_path):
        return self.h5_to_dict(file_path)
    
    
    def h5_to_dict(self, file_path):
        h5f = h5py.File(file_path, 'r')
        data_dict = {
            'image': torch.from_numpy(h5f['image'][:]), 
            'label': torch.from_numpy(h5f['label'][:])
        }
        h5f.close()
        return data_dict