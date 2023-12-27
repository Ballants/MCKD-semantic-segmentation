import copy
import os
import time
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import MulticlassJaccardIndex
from transformers import OneFormerProcessor, AutoModelForUniversalSegmentation

from model.encoder_decoder import SwinDeepLabV3Plus
from utils.losses import DiceLoss, CrossEntropyFocalLoss
from utils.visualization import visualize_segmentation

warnings.simplefilter('ignore')

def teacher_forward(teacher, **inputs):
    raw_out = teacher(**inputs)

    class_queries_logits = raw_out.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = raw_out.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation_logits = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)  # not probability
    # print("sum: ", torch.sum(segmentation_logits[:, :, 0, 0], dim=1))
    segmentation_logits = F.interpolate(segmentation_logits, size=(128, 128),
                                        mode='bilinear', align_corners=False)

    semantic_segmentation = segmentation_logits.softmax(dim=1)

    semantic_segmentation = semantic_segmentation.argmax(dim=1)
    # print("segmentation_logits: ", segmentation_logits.shape)
    # print("semantic_segmentation: ", semantic_segmentation.shape)
    return segmentation_logits, semantic_segmentation


def train(stud_id, dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()

    path_to_save_model = f'../checkpoints/{dataset}/'
    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)
    checkpoint_path = path_to_save_model + f'student{stud_id}_ckpt.pth'

    match dataset:
        case "coco":
            num_classes = 133
            processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
            teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)

        case "cityscapes":
            num_classes = 19
            processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
            teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large").to(
                device)

    student = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)

    # freezing backbone's parameters
    for param in student.backbone.parameters():
        param.requires_grad = False

    # Load dataloaders
    train_dl = torch.load(f'../data_loaders/{dataset}/Train_dl_{stud_id}.pt')
    val_dl = torch.load(f'../data_loaders/{dataset}/Validation_dl_{stud_id}.pt')
    dls = {"train": train_dl, "val": val_dl}

    # Weights sum up to 1
    # todo
    kl_loss_weight = 0
    ce_loss_weight = 0
    dice_loss_weight = 0
    T = 10  # temperature

    match dataset:
        case "coco":
            learning_rate = 1e-03  # learning rate
            milestones = [10, 13, 16]  # the epochs after which the learning rate is adjusted by gamma
            gamma = 0.1  # gamma correction to the learning rate, after reaching the milestone epochs
            weight_decay = 1e-05  # weight decay (L2 penalty)
            epochs = 20

        case "cityscapes":
            learning_rate = 1e-03  # learning rate
            milestones = [8, 11, 15]  # the epochs after which the learning rate is adjusted by gamma
            gamma = 0.1  # gamma correction to the learning rate, after reaching the milestone epochs
            weight_decay = 1e-05  # weight decay (L2 penalty)
            epochs = 20

    patience = 4  # Number of epochs with no improvement after which training will be stopped [early stopping]
    optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_scheduler = True  # use MultiStepLR scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    ce_loss = nn.CrossEntropyLoss()
    # ce_loss = CrossEntropyFocalLoss()  # focal loss
    dice_loss = DiceLoss()

    jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    teacher.eval()  # Teacher set to evaluation mode

    losses = {'train': [], 'val': []}
    val_iou_history = []
    val_f1_history = []

    single_losses = {'kl_l_train': [], 'dice_l_train': [], 'ce_l_train': [],
                     'kl_l_val': [], 'dice_l_val': [], 'ce_l_val': []}

    since = time.time()

    best_iou = 0.0
    best_f1 = 0.0
    best_val_loss = float('inf')
    current_patience = 0

    for epoch in range(epochs):
        for phase in ['train', 'val']:

            if phase == 'train':
                student.train()
            else:
                student.eval()

            running_loss = 0
            jaccard_index = 0
            f1_score = 0
            running_kl = 0
            running_dice = 0
            running_ce = 0
            n = 0
            for i, (images, _) in enumerate(dls[phase]):
                batch_size = images.shape[0]
                n += batch_size

                semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                    do_rescale=False).to(device)
                semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)

                match dataset:
                    case "coco":
                        student_input = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                                      mode='bilinear', align_corners=False)
                    case "cityscapes":
                        semantic_inputs["pixel_values"] = F.interpolate(semantic_inputs["pixel_values"],
                                                                        size=(512, 512),
                                                                        mode='bilinear', align_corners=False)
                        semantic_inputs["pixel_mask"] = F.interpolate(
                            semantic_inputs["pixel_mask"].type(torch.float32).unsqueeze(0), size=(512, 512),
                            mode='bilinear', align_corners=False).squeeze(0)
                        student_input = semantic_inputs["pixel_values"]

                optimizer.zero_grad()

                # Forward pass of the teacher model
                with torch.no_grad():
                    teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)
                # teacher_logits: [batch_size, num_classes, 128, 128]
                # pseudo_labels: [batch_size, 128, 128]

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass of the student model
                    student_logits, preds = student(student_input)
                    # student_logits: [batch_size, num_classes, 128, 128]
                    # preds: [batch_size, 128, 128]

                    # Soften the student logits by applying softmax first and log() second
                    soft_targets = F.softmax(teacher_logits / T, dim=1)
                    soft_prob = F.log_softmax(student_logits / T, dim=1)

                    # [batch_size * width * height, classes]
                    soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, num_classes)
                    soft_prob = soft_prob.permute(0, 2, 3, 1).reshape(-1, num_classes)

                    # Calculate the soft targets' loss. ["Distilling the knowledge in a neural network"]
                    kl_div_res = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)

                    # Calculate the true label loss
                    ce_res = ce_loss(student_logits, pseudo_labels)

                    # Calculate the true label loss
                    dice_res = dice_loss(student_logits, pseudo_labels)

                    # Weighted sum of the two losses
                    loss = kl_loss_weight * kl_div_res + ce_loss_weight * ce_res + dice_loss_weight * dice_res

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                jaccard_index += jaccard(preds, pseudo_labels) * batch_size
                f1_score += multiclass_f1_score(preds.reshape(-1), pseudo_labels.reshape(-1),
                                                num_classes=num_classes, average="weighted") * batch_size
                running_loss += loss * batch_size
                running_kl += kl_loss_weight * kl_div_res * batch_size
                running_ce += ce_loss_weight * ce_res * batch_size
                running_dice += dice_loss_weight * dice_res * batch_size

            epoch_loss = running_loss.detach().cpu() / n
            losses[phase].append(epoch_loss)

            single_losses['kl_l_' + phase].append(running_kl.detach().cpu() / n)
            single_losses['dice_l_' + phase].append(running_dice.detach().cpu() / n)
            single_losses['ce_l_' + phase].append(running_ce.detach().cpu() / n)

            epoch_jaccard_index = jaccard_index.detach().cpu() / n
            epoch_f1_score = f1_score.detach().cpu() / n

            if phase == 'train':
                print('Epoch: {}/{}'.format(epoch + 1, epochs))
            print('{}:\n- loss: {}\n- iou: {}\n- f1_score: {}'.format(phase, epoch_loss, epoch_jaccard_index,
                                                                      epoch_f1_score))

            if phase == 'val':
                print('Time: {}m {}s'.format((time.time() - since) // 60, (time.time() - since) % 60))
                val_iou_history.append(epoch_jaccard_index)
                val_f1_history.append(epoch_f1_score)
                if epoch_jaccard_index > best_iou:
                    best_iou = epoch_jaccard_index
                    best_model_state_dict = copy.deepcopy(student.state_dict())
                    # torch.save(best_model_state_dict, checkpoint_path)
                if epoch_f1_score > best_f1:
                    best_f1 = epoch_f1_score
                    best_model_state_dict = copy.deepcopy(student.state_dict())
                    # torch.save(best_model_state_dict, checkpoint_path)
                # early stopping
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    current_patience = 0
                else:
                    current_patience += 1
        if current_patience >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if use_scheduler:
            scheduler.step()

        # Plot
        if (epoch + 1) % 1 == 0:

            # Train and validation losses
            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), losses["train"], label='Training Loss', marker='o')
            plt.plot(range(epoch + 1), losses["val"], label='Validation Loss', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)
            # Save plot
            loss_plot_path = f'{dataset}_loss_stud_{stud_id}_epoch_{epoch + 1}.png'
            plt.savefig(loss_plot_path)
            plt.show()

            # Train and validation single losses
            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), single_losses['kl_l_train'], label='Train KL Loss', marker='o')
            plt.plot(range(epoch + 1), single_losses['kl_l_val'], label='Val KL Loss', marker='o')
            plt.plot(range(epoch + 1), single_losses['dice_l_train'], label='Train Dice Loss', marker='o')
            plt.plot(range(epoch + 1), single_losses['dice_l_val'], label='Val Dice Loss', marker='o')
            plt.plot(range(epoch + 1), single_losses['ce_l_train'], label='Train Focal Loss', marker='o')
            plt.plot(range(epoch + 1), single_losses['ce_l_val'], label='Val Focal Loss', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)
            # Save plot
            loss_plot_path = f'{dataset}_loss_all_stud_{stud_id}_epoch_{epoch + 1}.png'
            plt.savefig(loss_plot_path)
            plt.show()

            # Metrics
            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), val_iou_history, label='Jaccard index', marker='o')
            plt.plot(range(epoch + 1), val_f1_history, label='F1 score', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Validation Metrics')
            plt.legend()
            plt.grid(True)
            # Save plot
            loss_plot_path = f'{dataset}_metrics_stud_{stud_id}_epoch_{epoch + 1}.png'
            plt.savefig(loss_plot_path)
            plt.show()

            # show 2 examples of (teacher_pseudo_labels, student_predictions)
            visualize_segmentation(pseudo_labels[0], dataset)
            visualize_segmentation(preds[0], dataset)
            visualize_segmentation(pseudo_labels[1], dataset)
            visualize_segmentation(preds[1], dataset)

    time_elapsed = time.time() - since
    print('Training completed in: {}m {}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation iou: {}'.format(best_iou))


def test(stud_id, dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    path_to_model = f'../checkpoints/{dataset}/student{stud_id}_ckpt.pth'
    match dataset:
        case "coco":
            num_classes = 133
            processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
            teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)

        case "cityscapes":
            num_classes = 19
            processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
            teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large").to(
                device)

    student = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)
    student.load_state_dict(torch.load(path_to_model))

    # Load test dataloader
    test_dl = torch.load(f'../data_loaders/{dataset}/Test_dl.pt')

    # Set metrics
    jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    student.eval()

    with torch.no_grad():
        n = 0

        jaccard_index = 0
        f1_score = 0

        for i, (images, _) in enumerate(test_dl):
            batch_size = images.shape[0]
            n += batch_size

            semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                do_rescale=False).to(device)
            semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)

            match dataset:
                case "coco":
                    student_input = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                                  mode='bilinear', align_corners=False)
                case "cityscapes":
                    semantic_inputs["pixel_values"] = F.interpolate(semantic_inputs["pixel_values"],
                                                                    size=(512, 512),
                                                                    mode='bilinear', align_corners=False)
                    semantic_inputs["pixel_mask"] = F.interpolate(
                        semantic_inputs["pixel_mask"].type(torch.float32).unsqueeze(0), size=(512, 512),
                        mode='bilinear', align_corners=False).squeeze(0)
                    student_input = semantic_inputs["pixel_values"]

            teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)

            student_logits, preds = student(student_input)

            jaccard_index += jaccard(preds, pseudo_labels) * batch_size
            f1_score += multiclass_f1_score(preds.reshape(-1), pseudo_labels.reshape(-1),
                                            num_classes=num_classes, average="weighted") * batch_size

        jaccard_index = jaccard_index.detach().cpu() / n
        f1_score = f1_score.detach().cpu() / n

    print(f"Jaccard Index: {jaccard_index}")
    print(f"F1 Score: {f1_score}")

    # show 2 examples of (teacher_pseudo_labels, student_predictions)
    visualize_segmentation(pseudo_labels[0], dataset, path_to_save=f'{dataset}_stud_{stud_id}_test1_pseudo_label.png')
    visualize_segmentation(preds[0], dataset, path_to_save=f'{dataset}_stud_{stud_id}_test1_pred.png')

    visualize_segmentation(pseudo_labels[1], dataset, path_to_save=f'{dataset}_stud_{stud_id}_test2_pseudo_label.png')
    visualize_segmentation(preds[1], dataset, path_to_save=f'{dataset}_stud_{stud_id}_test2_pred.png')


if __name__ == '__main__':
    for dataset in ["coco", "cityscapes"]:
        # Student 1
        print(f'### Starting training student1 on {dataset} dataset... ###')
        train(stud_id=1, dataset=dataset)
        print(f'### Testing student1 on {dataset} dataset... ###')
        test(stud_id=1, dataset=dataset)

        # Student 2
        print(f'### Starting training student2 on {dataset} dataset... ###')
        train(stud_id=2, dataset=dataset)
        print(f'### Testing student2 on {dataset} dataset... ###')
        test(stud_id=2, dataset=dataset)
