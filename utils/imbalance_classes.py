import json
import os

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from transformers import OneFormerProcessor, AutoModelForUniversalSegmentation

from constants import id2label_coco


def teacher_forward(teacher, **inputs):
    raw_out = teacher(**inputs)

    class_queries_logits = raw_out.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = raw_out.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)  # not probability
    segmentation_logits = F.interpolate(segmentation_logits, size=(128, 128),
                                        mode='bilinear', align_corners=False)

    semantic_segmentation = segmentation_logits.softmax(dim=1)

    semantic_segmentation = semantic_segmentation.argmax(dim=1)
    return segmentation_logits, semantic_segmentation


def check_imbalance(dl_path, json_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    torch.cuda.empty_cache()

    processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)

    # Load dataloaders
    train_dl = torch.load(dl_path)

    teacher.eval()  # Teacher set to evaluation mode

    class_counts = {class_id: 0 for class_id in range(133)}

    for i, (images, _) in enumerate(train_dl):
        print("i: ", i)

        batch_size = images.shape[0]

        semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                            do_rescale=False).to(device)
        semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)

        with torch.no_grad():
            _, pseudo_labels = teacher_forward(teacher, **semantic_inputs)

        # Iterate through each mask in the batch
        for mask in pseudo_labels:
            # Iterate through unique class IDs in the mask
            unique_classes = set(mask.flatten().cpu().numpy().astype(int))
            for class_id in unique_classes:
                # Increment the count for the current class
                class_counts[class_id] += 1

    for class_id, count in class_counts.items():
        print(f"Class {class_id}: {count} images")

    class_counts_json = json.dumps(class_counts, indent=2)
    with open(json_path, 'w') as json_file:
        json_file.write(class_counts_json)


def plot_histogram(json_path, hist_path):
    with open(json_path, 'r') as json_file:
        class_counts = json.load(json_file)

    # Extract class IDs and counts
    class_ids = list(class_counts.keys())
    class_labels = [id2label_coco[int(i)] for i in class_ids]
    counts = list(class_counts.values())

    plt.figure(figsize=(20, 10))
    plt.bar(class_labels, counts, color='blue')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in the Dataset')

    plt.xticks(rotation=90, fontsize=8)

    plt.tight_layout()
    plt.margins(x=0.007)

    plt.savefig(hist_path)

    plt.show()


if __name__ == '__main__':

    d = "../info_coco/"
    if not os.path.exists(d):
        os.makedirs(d)

    check_imbalance(dl_path="../data_loaders/coco/Train_dl_1.pt",
                    json_path="../info_coco/num_imgs_x_class_stud1.json")
    plot_histogram(json_path="../info_coco/num_imgs_x_class_stud1.json",
                   hist_path="../info_coco/num_imgs_x_class_stud1_hist.png")

    check_imbalance(dl_path="../data_loaders/coco/Train_dl_2.pt",
                    json_path="../info_coco/num_imgs_x_class_stud2.json")
    plot_histogram(json_path="../info_coco/num_imgs_x_class_stud2.json",
                   hist_path="../info_coco/num_imgs_x_class_stud2_hist.png")

    check_imbalance(dl_path="../data_loaders/coco/Train_dl.pt",
                    json_path="../info_coco/num_imgs_x_class.json")
    plot_histogram(json_path="../info_coco/num_imgs_x_class.json",
                   hist_path="../info_coco/num_imgs_x_class_hist.png")
