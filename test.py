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
index = '00001'
label_path = glob(r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData\BraTS-GoAT-' + index + '\*seg*')
image_path = [file_dir for file_dir in glob(r'E:\SPRV_Brain\data\ISBI2024-BraTS-GoAT-TrainingData\ISBI2024-BraTS-GoAT-TrainingData\BraTS-GoAT-' + index + '\*') if file_dir not in label_path]
print(image_path)
data_dict = {'image': image_path,
             'label': label_path}
print(data_dict)
data_reader = transforms.LoadImaged(keys = ["image", "label"])

data = data_reader(data_dict)


img, label = data['image'], data['label']

label = (label != 0).to(label.dtype)


meta_dict = data['image'].meta
# print(meta_dict.keys(), meta_dict)
print(meta_dict['pixdim'])


print(img.shape)

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'image' is your image data, a 2D numpy array
# Replace it with your actual image data

img = img[2, :, :, 45]
# Compute histogram of pixel intensities
hist, bar = np.histogram(img.flatten(), bins=1000, range=(300, 2000))

# Plot the histogram
# plt.bar(bar[:-1], hist)
# plt.show()


# plt.imshow(label[:, :, 45])
# plt.show()
