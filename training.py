import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import MulticlassJaccardIndex
from transformers import OneFormerProcessor, AutoModelForUniversalSegmentation

from model.encoder_decoder import SwinDeepLabV3Plus
from utils.losses import DiceLoss
from utils.visualization import visualize_segmentation

# todo hyper-parameters
learning_rate = 1e-03  # learning rate
milestones = [2, 4, 6, 10]  # the epochs after which the learning rate is adjusted by gamma
gamma = 0.1  # gamma correction to the learning rate, after reaching the milestone epochs
weight_decay = 1e-05  # weight decay (L2 penalty)
epochs = 20
T = 10  # Temperature

# Weights sum up to 1
# TODO paper 2015 dice che deve essere più alto al resto... online è sempre più basso del resto (tipo regularization)
kl_loss_weight = 0.2  # 0.5
ce_loss_weight = 0.8  # 0.5 * 0.25
dice_loss_weight = 0.0  # 0.5 * 0.75


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


def train(stud_id, path_to_save_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()

    '''processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)'''

    processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
    teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large").to(device)

    student = SwinDeepLabV3Plus(num_classes=19).to(device)

    # freezing backbone's parameters
    for param in student.backbone.parameters():
        param.requires_grad = False

    # Load dataloaders
    train_dl = torch.load(f'./data_loaders/cityscapes/Train_dl_{stud_id}.pt')
    val_dl = torch.load(f'./data_loaders/cityscapes/Validation_dl_{stud_id}.pt')
    '''train_dl = torch.load(f'./data_loaders/Test_dl_{1}.pt')
    val_dl = torch.load(f'./data_loaders/Test_dl_{2}.pt')'''

    optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_scheduler = True  # use MultiStepLR scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    ce_loss = nn.CrossEntropyLoss()  # todo usiamo focal al posto suo?
    # ce_loss = CrossEntropyFocalLoss()  # focal loss
    dice_loss = DiceLoss()

    jaccard = MulticlassJaccardIndex(num_classes=19).to(device)

    # Todo training loop
    teacher.eval()  # Teacher set to evaluation mode

    train_loss = []
    val_loss = []

    kl_l_t = []
    dice_l_t = []
    focal_l_t = []

    kl_l_v = []
    dice_l_v = []
    focal_l_v = []

    jaccard_index_list = []
    f1_score_list = []

    ## Training ##
    for epoch in range(epochs):

        student.train()
        running_loss = 0
        running_kl = 0
        running_dice = 0
        running_focal = 0
        n = 0
        for i, (images, _) in enumerate(train_dl):
            print("# epoch: ", epoch, " - i: ", i)

            batch_size = images.shape[0]
            n += batch_size

            semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                do_rescale=False).to(device)
            semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)
            # print("pixel_values: ", semantic_inputs['pixel_values'].shape)

            semantic_inputs["pixel_values"] = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                                            mode='bilinear', align_corners=False)
            '''semantic_inputs["pixel_mask"] = F.interpolate(semantic_inputs["pixel_mask"].unsqueeze(0), size=(512, 512),
                                          mode='bilinear', align_corners=False)'''
            semantic_inputs["pixel_mask"] = F.interpolate(
                semantic_inputs["pixel_mask"].type(torch.float32).unsqueeze(0), size=(512, 512), mode='bilinear',
                align_corners=False).squeeze(0)

            optimizer.zero_grad()

            # Forward pass of the teacher model - do not save gradients to not change the teacher's weights
            with torch.no_grad():
                teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)
            # print("teacher_logits: ", teacher_logits.shape)
            # print("pseudo_labels: ", pseudo_labels.shape)

            # Forward pass of the student model
            student_logits, preds = student(semantic_inputs["pixel_values"])
            # print("student_logits: ", student_logits.shape)  # not probability
            # print("preds: ", preds.shape)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = F.softmax(teacher_logits / T, dim=1)
            soft_prob = F.log_softmax(student_logits / T, dim=1)

            # [batch * width * height, classes]
            soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, 19)
            soft_prob = soft_prob.permute(0, 2, 3, 1).reshape(-1, 19)

            # Calculate the soft targets loss. ["Distilling the knowledge in a neural network"]
            kl_div_res = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)

            # Calculate the true label loss
            ce_res = ce_loss(student_logits, pseudo_labels)

            # Calculate the true label loss
            dice_res = dice_loss(student_logits, pseudo_labels)

            # Weighted sum of the two losses
            loss = kl_loss_weight * kl_div_res + ce_loss_weight * ce_res + dice_loss_weight * dice_res

            loss.backward()

            optimizer.step()
            running_loss += loss * batch_size
            running_kl += kl_loss_weight * kl_div_res * batch_size
            running_dice += dice_loss_weight * dice_res * batch_size
            running_focal += ce_loss_weight * ce_res * batch_size

        train_loss.append(running_loss.detach().cpu() / n)
        kl_l_t.append(running_kl.detach().cpu() / n)
        dice_l_t.append(running_dice.detach().cpu() / n)
        focal_l_t.append(running_focal.detach().cpu() / n)

        ## Validation ##
        student.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0

            jaccard_index = 0
            f1_score = 0

            for i, (images, _) in enumerate(val_dl):

                batch_size = images.shape[0]
                n += batch_size

                semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                    do_rescale=False).to(device)
                semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)
                semantic_inputs["pixel_values"] = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                                                mode='bilinear', align_corners=False)
                semantic_inputs["pixel_mask"] = F.interpolate(
                    semantic_inputs["pixel_mask"].type(torch.float32).unsqueeze(0), size=(512, 512), mode='bilinear',
                    align_corners=False).squeeze(0)

                teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)

                student_logits, preds = student(semantic_inputs["pixel_values"])

                soft_targets = F.softmax(teacher_logits / T, dim=1)
                soft_prob = F.log_softmax(student_logits / T, dim=1)

                # [batch * width * height, classes]
                soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, 19)
                soft_prob = soft_prob.permute(0, 2, 3, 1).reshape(-1, 19)

                kl_div_res = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)
                ce_res = ce_loss(student_logits, pseudo_labels)
                dice_res = dice_loss(student_logits, pseudo_labels)

                jaccard_index += jaccard(preds, pseudo_labels) * batch_size
                f1_score += multiclass_f1_score(preds.reshape(-1), pseudo_labels.reshape(-1),
                                                num_classes=19, average="weighted") * batch_size

                loss = kl_loss_weight * kl_div_res + ce_loss_weight * ce_res + dice_loss_weight * dice_res

                running_loss += loss * batch_size
                running_kl += kl_loss_weight * kl_div_res * batch_size
                running_dice += dice_loss_weight * dice_res * batch_size
                running_focal += ce_loss_weight * ce_res * batch_size

            val_loss.append(running_loss.detach().cpu() / n)
            kl_l_v.append(running_kl.detach().cpu() / n)
            dice_l_v.append(running_dice.detach().cpu() / n)
            focal_l_v.append(running_focal.detach().cpu() / n)
            jaccard_index_list.append(jaccard_index.detach().cpu() / n)
            f1_score_list.append(f1_score.detach().cpu() / n)

        if use_scheduler:
            scheduler.step()

        # Save and plot
        if (epoch + 1) % 1 == 0:  # todo %10

            if path_to_save_model is not None and (epoch + 1) % 5 == 0:
                checkpoint_path = path_to_save_model + f'student_ckpt_epoch_{epoch + 1}.pth'
                torch.save(student.state_dict(), checkpoint_path)

            plt.figure(figsize=(10, 6))
            # plt.plot(range(epoch + 1), train_loss, label='Training Loss', marker='o')
            # plt.plot(range(epoch + 1), val_loss, label='Validation Loss', marker='o')

            plt.plot(range(epoch + 1), kl_l_t, label='Train KL Loss', marker='o')
            plt.plot(range(epoch + 1), kl_l_v, label='Val KL Loss', marker='o')
            plt.plot(range(epoch + 1), dice_l_t, label='Train Dice Loss', marker='o')
            plt.plot(range(epoch + 1), dice_l_v, label='Val Dice Loss', marker='o')
            plt.plot(range(epoch + 1), focal_l_t, label='Train Focal Loss', marker='o')
            plt.plot(range(epoch + 1), focal_l_v, label='Val Focal Loss', marker='o')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)

            # Save the loss plot
            '''loss_plot_path = path_to_save_model + f'loss_plot_epoch_{epoch + 1}.png'
            plt.savefig(loss_plot_path)'''

            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), jaccard_index_list, label='Jaccard index', marker='o')
            plt.plot(range(epoch + 1), f1_score_list, label='F1 score', marker='o')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Validation Metrics')
            plt.legend()
            plt.grid(True)

            plt.show()

            visualize_segmentation(pseudo_labels[0])
            visualize_segmentation(pseudo_labels[1])

            visualize_segmentation(preds[0])
            visualize_segmentation(preds[1])


# todo review
def test(stud_id, path_to_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)

    # processor_student = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224") # inutile
    student = SwinDeepLabV3Plus(num_classes=19).to(device)
    student.load_state_dict(torch.load(path_to_model))

    # freezing backbone's parameters
    for param in student.backbone.parameters():
        param.requires_grad = False

    # Load test dataloader
    test_dl = torch.load(f'./data_loaders/cityscapes/Test_dl_{stud_id}.pt')

    # Set metrics
    jaccard = MulticlassJaccardIndex(num_classes=19).to(device)

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
            student_input = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                          mode='bilinear', align_corners=False)
            teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)

            student_logits, preds = student(student_input)

            jaccard_index += jaccard(preds, pseudo_labels) * batch_size
            f1_score += multiclass_f1_score(preds.reshape(-1), pseudo_labels.reshape(-1),
                                            num_classes=19, average="weighted") * batch_size

        jaccard_index = jaccard_index.detach().cpu() / n
        f1_score = f1_score.detach().cpu() / n

    print(f"Jaccard Index: {jaccard_index}")
    print(f"F1 Score: {f1_score}")

    visualize_segmentation(pseudo_labels[0], path_to_save="lab_1_stud_"+stud_id+".png")
    visualize_segmentation(pseudo_labels[1], path_to_save="lab_2_stud_"+stud_id+".png")

    visualize_segmentation(preds[0], path_to_save="pred_1_stud_"+stud_id+".png")
    visualize_segmentation(preds[1], path_to_save="pred_2_stud_"+stud_id+".png")


if __name__ == '__main__':
    path_to_save_model = "./checkpoints/cityscapes/"
    print("### Starting training... ###")
    t0 = time.time()
    train(stud_id=1, path_to_save_model=path_to_save_model)
    t1 = time.time()
    print("training/validation time: {0:.2f}s".format(t1 - t0))

    t2 = time.time()
    test(stud_id=1, path_to_model=path_to_save_model + "cityscapes_stud_1.pth")
    t3 = time.time()
    print("training/validation time for the final student: {0:.2f}s".format(t3 - t2))
