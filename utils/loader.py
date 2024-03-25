import h5py
import torch
from monai.transforms import Transform
from monai.data import Dataset, DataLoader




def get_loader_seg(data_dir_list: list, reader: Transform, batch_size: int, shuffle: bool, drop_last: bool, num_workers=0):
    dataset = Dataset(dataset=data_dir_list, transform=reader)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_loader_ranking(data_dir_list: list, reader: Transform, batch_size: int, sequence_length: int, shuffle: bool, drop_last: bool, num_workers=0):
    dataset = Dataset(dataset=data_dir_list, transform=reader)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size*sequence_length, shuffle=shuffle, drop_last=drop_last)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.file_loader import ReadH5d
    reader = ReadH5d()
    data_dict = reader(r'E:\SPRV_Brain\data\BraTs_H5\BarTs00302.h5')
    
    img, label = data_dict['image'], data_dict['label']
    # Generate some random data for plotting
    data = img[0]

    # Create a figure and an array of subplots
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))

    # Flatten the array of subplots for easier iteration
    axes = axes.flatten()

    # Iterate through the subplots and plot data
    for i, ax in enumerate(axes):
        ax.imshow(data[:, :, i], cmap='viridis')  # Change the colormap as needed
        ax.set_title(f'Subplot {i+1}')
        ax.axis('off')  # Turn off axis for clarity

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()

        