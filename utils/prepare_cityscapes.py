import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode

BATCH_SIZE = 8


def plot_random_samples(data_loader):
    # Create a figure with a 3x3 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # Iterate through the DataLoader to get one batch
    for batch in data_loader:
        # Extract the first 4 samples from the batch
        samples = batch[0][:4]
        # print(samples.shape)

        # Display each tensor as an image in the grid
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                img = samples[idx].permute(1, 2, 0).numpy()  # Assuming tensors are in CHW format
                img = np.clip(img, 0, 1)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')

        plt.suptitle("Sample images from dataset", fontsize=26)

        plt.tight_layout()
        plt.show()

        # Break after the first batch to only visualize one batch
        break


def prepare_data():
    data_transform = transforms.Compose([transforms.Resize(224),  # resize shortest side
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
    target_transforms = Compose([
        transforms.Resize(size=224, interpolation=InterpolationMode.NEAREST),  # resize shortest side
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = Cityscapes(root='../datasets/cityscapes_dataset/',
                          split="train",
                          target_type="semantic",
                          mode="fine",
                          transform=data_transform,
                          target_transform=target_transforms)

    val_ds = Cityscapes(root='../datasets/cityscapes_dataset/',
                        split="val",
                        target_type="semantic",
                        mode="fine",
                        transform=data_transform,
                        target_transform=target_transforms)

    test_ds = Cityscapes(root='../datasets/cityscapes_dataset/',
                         split="test",
                         target_type="semantic",
                         mode="fine",
                         transform=data_transform,
                         target_transform=target_transforms)

    print('Len Training dataset: ', len(train_ds))  # 2975
    print('Len Validation dataset: ', len(val_ds))  # 500
    print('Len Testing dataset: ', len(test_ds))  # 1525

    train_ds_1, train_ds_2 = train_test_split(train_ds, train_size=0.5, shuffle=True, random_state=42)
    val_ds_1, val_ds_2 = train_test_split(val_ds, train_size=0.5, shuffle=True, random_state=42)
    test_ds_1, test_ds_2 = train_test_split(test_ds, train_size=0.5, shuffle=True, random_state=42)
    print('Len Training dataset_1: ', len(train_ds_1))  # 1488
    print('Len Training dataset_2: ', len(train_ds_2))  # 1487
    print('Len Validation dataset_1: ', len(val_ds_1))  # 250
    print('Len Validation dataset_2: ', len(val_ds_2))  # 250
    print('Len Test dataset 1: ', len(test_ds_1))       # 763
    print('Len Test dataset 1: ', len(test_ds_2))       # 762

    train_dl_1 = torch.utils.data.DataLoader(train_ds_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    train_dl_2 = torch.utils.data.DataLoader(train_ds_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dl_1 = torch.utils.data.DataLoader(val_ds_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dl_2 = torch.utils.data.DataLoader(val_ds_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dl_1 = torch.utils.data.DataLoader(test_ds_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dl_2 = torch.utils.data.DataLoader(test_ds_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Plot few samples
    plot_random_samples(train_dl_1)

    dataloader_dir = "../data_loaders/cityscapes/"
    if not os.path.exists(dataloader_dir):
        os.makedirs(dataloader_dir)

    # Saving Dataloaders
    torch.save(train_dl_1, dataloader_dir + 'Train_dl_1.pt')
    torch.save(train_dl_2, dataloader_dir + 'Train_dl_2.pt')
    torch.save(val_dl_1, dataloader_dir + 'Validation_dl_1.pt')
    torch.save(val_dl_2, dataloader_dir + 'Validation_dl_2.pt')
    torch.save(test_dl_1, dataloader_dir + 'Test_dl_1.pt')
    torch.save(test_dl_2, dataloader_dir + 'Test_dl_2.pt')


if __name__ == '__main__':
    print("### Preparing DataLoader... ###")
    prepare_data()
    print("### DataLoader ready ###")
