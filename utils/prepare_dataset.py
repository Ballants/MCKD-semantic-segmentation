import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

BATCH_SIZE = 8
PERC_DATA = 0.1  # percentage of dataset we are using


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

    # transforms (callable, optional) – A function/transform that takes input sample and its target as entry and returns a transformed version.
    # TODO fun fact: nella versione "test2017" (41k imgs, 6GB) non ci sono le annotazioni
    coco_dataset = torchvision.datasets.CocoDetection(root="../coco_dataset/test2017",
                                                      annFile="../coco_dataset/annotations/image_info_test2017.json",
                                                      transform=data_transform)  # Todo check if "transforms" works on target data (check resulting images)

    print('\nLen dataset: ', len(coco_dataset))  # 40670

    ds, _ = train_test_split(coco_dataset, train_size=PERC_DATA, shuffle=True, random_state=42)

    train_ds, val_test_ds = train_test_split(ds, train_size=0.7, shuffle=True, random_state=42)
    print('Len Training dataset: ', len(train_ds))  # 28469     #
    val_ds, test_ds = train_test_split(val_test_ds, test_size=1 / 3, shuffle=True, random_state=42)
    print('Len Validation dataset: ', len(val_ds))  # 8134     #
    print('Len Testing dataset: ', len(test_ds))  # 4067     #

    train_ds_1, train_ds_2 = train_test_split(train_ds, train_size=0.5, shuffle=True, random_state=42)
    val_ds_1, val_ds_2 = train_test_split(val_ds, train_size=0.5, shuffle=True, random_state=42)
    test_ds_1, test_ds_2 = train_test_split(test_ds, train_size=0.5, shuffle=True, random_state=42)
    print('Len Training dataset_1: ', len(train_ds_1))  # 14234     #
    print('Len Training dataset_2: ', len(train_ds_2))  # 14235     #
    print('Len Validation dataset_1: ', len(val_ds_1))  # 4067     #
    print('Len Validation dataset_2: ', len(val_ds_2))  # 4067     #
    print('Len Testing dataset_1: ', len(test_ds_1))  # 2033     #
    print('Len Testing dataset_2: ', len(test_ds_2))  # 2034     #

    train_dl_1 = torch.utils.data.DataLoader(train_ds_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    train_dl_2 = torch.utils.data.DataLoader(train_ds_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dl_1 = torch.utils.data.DataLoader(val_ds_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_dl_2 = torch.utils.data.DataLoader(val_ds_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dl_1 = torch.utils.data.DataLoader(test_ds_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dl_2 = torch.utils.data.DataLoader(test_ds_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Plot few samples
    plot_random_samples(train_dl_1)

    dataloader_dir = "../data_loaders/"
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
