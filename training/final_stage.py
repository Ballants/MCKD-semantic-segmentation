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
from transformers import OneFormerProcessor

from model.encoder_decoder import SwinDeepLabV3Plus
from utils.losses import DiceLoss, CrossEntropyFocalLoss
from utils.visualization import visualize_segmentation

warnings.simplefilter('ignore')

def student_inputs(images, device, processor):
    semantic_inputs = processor(images=images, task_inputs=["semantic"], return_tensors="pt",
                                do_rescale=False).to(device)
    student_input = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                  mode='bilinear', align_corners=False)
    return student_input


def final_train(dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    path_to_save_model = f'../checkpoints/{dataset}/'
    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)
    checkpoint_path = path_to_save_model + f'final_student_ckpt.pth'

    match dataset:
        case "coco":
            num_classes = 133
            processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

        case "cityscapes":
            num_classes = 19
            processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")

    student1 = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)  # --> train su Da --> pred Db
    student2 = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)  # --> train Db --> pred su Da
    student1.load_state_dict(torch.load(path_to_save_model + 'student1_ckpt.pth'))
    student2.load_state_dict(torch.load(path_to_save_model + 'student2_ckpt.pth'))
    student1.eval()
    student2.eval()
    teachers = [student1, student2]

    train_dl_stud1 = torch.load(f'../data_loaders/{dataset}/Train_dl_2.pt')
    train_dl_stud2 = torch.load(f'../data_loaders/{dataset}/Train_dl_1.pt')
    val_dl_stud1 = torch.load(f'../data_loaders/{dataset}/Validation_dl_1.pt')
    val_dl_stud2 = torch.load(f'../data_loaders/{dataset}/Validation_dl_2.pt')
    dls = {"train": [train_dl_stud1, train_dl_stud2], "val": [val_dl_stud1, val_dl_stud2]}

    final_student = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)
    # freezing backbone's parameters
    for param in final_student.backbone.parameters():
        param.requires_grad = False

    # Weights sum up to 1
    # todo
    kl_loss_weight = 0.0
    ce_loss_weight = 0.0
    dice_loss_weight = 0.0
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
    optimizer = optim.Adam(final_student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_scheduler = True  # use MultiStepLR scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    ce_loss = nn.CrossEntropyLoss()
    # ce_loss = CrossEntropyFocalLoss()  # focal loss
    dice_loss = DiceLoss()
    jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)

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
                final_student.train()
            else:
                final_student.eval()

            running_loss = 0
            jaccard_index = 0
            f1_score = 0
            running_kl = 0
            running_dice = 0
            running_ce = 0
            n = 0
            for stud_id in range(2):
                for i, (images, _) in enumerate(dls[phase][stud_id]):
                    # Computational limit: train only on half of the dataloader
                    if phase == "train" and i == 85:
                        break

                    batch_size = images.shape[0]
                    n += batch_size

                    student_input = student_inputs(images, device, processor)

                    optimizer.zero_grad()

                    # Forward pass of the teacher model
                    with torch.no_grad():
                        teacher = teachers[stud_id]
                        teacher_logits, pseudo_labels = teacher(student_input)

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass of the student model
                        student_logits, preds = final_student(student_input)

                        # Soften the student logits by applying softmax first and log() second
                        soft_targets = F.softmax(teacher_logits / T, dim=1)
                        soft_prob = F.log_softmax(student_logits / T, dim=1)

                        # [batch * width * height, classes]
                        soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, 133)
                        soft_prob = soft_prob.permute(0, 2, 3, 1).reshape(-1, 133)

                        # Calculate the soft targets loss. ["Distilling the knowledge in a neural network"]
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

                    jaccard_index += jaccard(preds, pseudo_labels) * batch_size
                    f1_score += multiclass_f1_score(preds.reshape(-1), pseudo_labels.reshape(-1),
                                                    num_classes=num_classes, average="weighted") * batch_size
                    running_loss += loss * batch_size
                    running_kl += kl_loss_weight * kl_div_res * batch_size
                    running_ce += ce_loss_weight * ce_res * batch_size
                    running_dice += dice_loss_weight * dice_res * batch_size

            epoch_loss = running_loss.detach().cpu() / n
            losses[phase].append(epoch_loss)

            single_losses['kl_l_'+phase].append(running_kl.detach().cpu() / n)
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
                        best_model_state_dict = copy.deepcopy(final_student.state_dict())
                        torch.save(best_model_state_dict, checkpoint_path)
                    if epoch_f1_score > best_f1:
                        best_f1 = epoch_f1_score
                        best_model_state_dict = copy.deepcopy(final_student.state_dict())
                        torch.save(best_model_state_dict, checkpoint_path)
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
            loss_plot_path = path_to_save_model + f'{dataset}_loss_plot_final_stud_epoch_{epoch + 1}.png'
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
            loss_plot_path = path_to_save_model + f'{dataset}_loss_all_final_stud_epoch_{epoch + 1}.png'
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
            loss_plot_path = path_to_save_model + f'{dataset}_metrics_final_stud_epoch_{epoch + 1}.png'
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


def test(dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    path = f'../checkpoints/{dataset}/'
    checkpoint_path = path + 'final_student_ckpt.pth'
    match dataset:
        case "coco":
            num_classes = 133
            processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

        case "cityscapes":
            num_classes = 19
            processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")

    student1 = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)  # --> train su Da --> pred Db
    student2 = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)  # --> train Db --> pred su Da
    student1.load_state_dict(torch.load(path + 'student1_ckpt.pth'))
    student2.load_state_dict(torch.load(path + 'student2_ckpt.pth'))
    student1.eval()
    student2.eval()
    teachers = [student1, student2]

    final_student = SwinDeepLabV3Plus(num_classes=num_classes, dataset=dataset).to(device)
    final_student.load_state_dict(torch.load(checkpoint_path))
    final_student.eval()

    # Load test dataloader
    test_dl_1 = torch.load(f'../data_loaders/{dataset}/Test_dl_1.pt')
    test_dl_2 = torch.load(f'../data_loaders/{dataset}/Test_dl_2.pt')
    dl = [test_dl_1, test_dl_2]

    # Set metrics
    jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    with torch.no_grad():
        n = 0

        jaccard_index = 0
        f1_score = 0

        for stud_id in range(2):
            for i, (images, _) in enumerate(dl[stud_id]):
                if i == 90:
                    break

                batch_size = images.shape[0]
                n += batch_size

                student_input = student_inputs(images, device, processor)

                # Forward pass of the teacher model
                teacher = teachers[stud_id]
                teacher_logits, pseudo_labels = teacher(student_input)

                # Forward pass of the student model
                student_logits, preds = final_student(student_input)

                jaccard_index += jaccard(preds, pseudo_labels) * batch_size
                f1_score += multiclass_f1_score(preds.reshape(-1), pseudo_labels.reshape(-1),
                                                num_classes=num_classes, average="weighted") * batch_size

        jaccard_index = jaccard_index.detach().cpu() / n
        f1_score = f1_score.detach().cpu() / n

    print(f"Jaccard Index: {jaccard_index}")
    print(f"F1 Score: {f1_score}")

    # show 2 examples of (teacher_pseudo_labels, student_predictions)
    visualize_segmentation(pseudo_labels[0], dataset, path_to_save=f'{dataset}_final_stud_test1_pseudo_label.png')
    visualize_segmentation(preds[0], dataset, path_to_save=f'{dataset}_final_stud_test1_pred.png')

    visualize_segmentation(pseudo_labels[1], dataset, path_to_save=f'{dataset}_final_stud_test2_pseudo_label.png')
    visualize_segmentation(preds[1], dataset, path_to_save=f'{dataset}_final_stud_test2_pred.png')


if __name__ == '__main__':
    for dataset in ["coco", "cityscapes"]:
        # Final student
        print(f'### Starting training the final student on {dataset} dataset... ###')
        final_train(dataset=dataset)
        print(f'### Testing the final student on {dataset} dataset... ###')
        test(dataset=dataset)
