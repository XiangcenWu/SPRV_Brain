import os
import json


print(len(os.listdir(r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData')))



# def datafold_read(datalist, basedir, fold=0, key="training"):
#     with open(datalist) as f:
#         json_data = json.load(f)

#     json_data = json_data[key]

#     for d in json_data:
#         for k in d:
#             if isinstance(d[k], list):
#                 d[k] = [os.path.join(basedir, iv) for iv in d[k]]
#             elif isinstance(d[k], str):
#                 d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

#     tr = []
#     val = []
#     for d in json_data:
#         if "fold" in d and d["fold"] == fold:
#             val.append(d)
#         else:
#             tr.append(d)

#     return tr, val


# x = datafold_read(
#     r'E:\SPRV_Brain\brats21_folds.json',
#     r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData',
# )

# print(type(x), x[0][0])




from monai import transforms
from glob import glob
label_path = glob(r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData\BraTS-GoAT-00000\*seg*')
image_path = [file_dir for file_dir in glob(r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData\BraTS-GoAT-00000\*') if file_dir not in label_path]
data_dict = {'image': image_path,
             'label': label_path}
data_reader = transforms.LoadImaged(keys = ["image", "label"])

data = data_reader(data_dict)

print(data.keys())
img, label = data['image'], data['label']

print('hello')

meta_dict = data['image'].meta

print(meta_dict)

