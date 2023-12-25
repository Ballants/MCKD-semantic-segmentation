import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from utils.constants import id2label_cityscapes, id2label_coco


def visualize_segmentation(segmentation_tensor, dataset, path_to_save=None):
    match dataset:
        case "coco":
            id2label = id2label_coco
        case "cityscapes":
            id2label = id2label_cityscapes

    # get all the unique numbers
    labels_ids = torch.unique(segmentation_tensor).tolist()
    print(labels_ids)

    # Map ids with RGB colors
    coco_color_map = {id: cm.viridis(index / len(labels_ids)) for index, id in enumerate(labels_ids)}

    # Map the class indices to RGB colors using NumPy vectorized operations
    segmented_image = np.zeros((segmentation_tensor.shape[0], segmentation_tensor.shape[1], 4), dtype=np.float32)
    class_indices = segmentation_tensor.long().cpu().numpy()

    mask = np.isin(class_indices, list(coco_color_map.keys()))
    segmented_image[mask] = [coco_color_map[class_index] for class_index in class_indices[mask]]

    # Create legend labels based on id2label mapping
    legend_labels = [id2label[class_id] for class_id in labels_ids]

    # Display the segmented image with legend
    plt.figure(figsize=(9, 9))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.title('Segmentation Map')

    handles = [mpatches.Patch(color=coco_color_map[label_id], label=id2label[label_id]) for label_id in labels_ids]

    plt.legend(handles=handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(path_to_save, bbox_inches='tight')

    plt.show()
