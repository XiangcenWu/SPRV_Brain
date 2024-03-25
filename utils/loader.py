import h5py
import torch




class ReadH5d(object):
    """Convert ndarrays in sample to Tensors."""

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
    
    

    

def get_loader_seg(batch_size, data_dir):
    pass


def get_loader_ranking(batch_size, sequence_length, data_dir):
    pass



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    reader = ReadH5d()
    data_dict = reader(r'E:\SPRV_Brain\data\BraTs_H5\BarTs00002.h5')
    
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

        