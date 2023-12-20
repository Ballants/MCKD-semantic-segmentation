import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import MulticlassJaccardIndex
from transformers import OneFormerProcessor

from training import train
from model.encoder_decoder import SwinDeepLabV3Plus
from utils.losses import DiceLoss
from utils.visualization import visualize_segmentation

path_stud1 = "./checkpoints/student1/"
path_stud2 = "./checkpoints/student2/"


def student_inputs(images, device, processor):
    batch_size = images.shape[0]
    semantic_inputs = processor(images=images, task_inputs=["semantic"], return_tensors="pt",
                                do_rescale=False).to(device)
    semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)
    student_input = F.interpolate(semantic_inputs["pixel_values"], size=(512, 512),
                                  mode='bilinear', align_corners=False)
    return student_input


def final_train(path_to_save_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")

    student1 = SwinDeepLabV3Plus(num_classes=133).to(device)    #--> train su Da --> pred Db
    student2 = SwinDeepLabV3Plus(num_classes=133).to(device)    #--> train Db --> pred su Da

    student1.load_state_dict(torch.load(path_stud1 + 'student_ckpt_epoch_1.pth'))       # fissare l'epoca giusta
    student2.load_state_dict(torch.load(path_stud2 + 'student_ckpt_epoch_1.pth'))       # fissare l'epoca giusta

    student1.eval()
    student2.eval()

    dl_stud1 = torch.load(f'./data_loaders/Train_dl_2.pt')
    dl_stud2 = torch.load(f'./data_loaders/Train_dl_1.pt')

    final_student = SwinDeepLabV3Plus(num_classes=133).to(device)

    # freezing backbone's parameters
    for param in final_student.backbone.parameters():
        param.requires_grad = False

    learning_rate = 1e-03  # learning rate
    milestones = [5, 10, 15]  # the epochs after which the learning rate is adjusted by gamma
    gamma = 0.1  # gamma correction to the learning rate, after reaching the milestone epochs
    weight_decay = 1e-05  # weight decay (L2 penalty)
    epochs = 20
    T = 2  # Temperature

    optimizer = optim.Adam(final_student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    use_scheduler = True  # use MultiStepLR scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Weights sum up to 1
    # TODO paper 2015 dice che deve essere più alto al resto... online è sempre più basso del resto (tipo regularization)
    kl_loss_weight = 0.2
    ce_loss_weight = 0.4
    dice_loss_weight = 0.4

    ce_loss = nn.CrossEntropyLoss()  # todo usiamo focal al posto suo?
    # ce_loss = CrossEntropyFocalLoss()  # focal loss
    dice_loss = DiceLoss()
    jaccard = MulticlassJaccardIndex(num_classes=133)

    train_loss = []
    val_loss = []

    kl_l = []
    dice_l = []
    focal_l = []

    jaccard_index_list = []
    f1_score_list = []

    for epoch in range(epochs):

        final_student.train()
        running_loss = 0
        running_kl = 0
        running_dice = 0
        running_focal = 0
        n = 0
        for i, (images, _) in enumerate(dl_stud1):
            print("# epoch: ", epoch, " - i: ", i)
            batch_size = images.shape[0]
            n += batch_size

            student_input = student_inputs(images, device, processor)

            optimizer.zero_grad()

            # Forward pass of the teacher model - do not save gradients to not change the teacher's weights
            with torch.no_grad():
                pseudo_logits, pseudo_labels = student1(student_input)
            # print("teacher_logits: ", teacher_logits.shape)
            # print("pseudo_labels: ", pseudo_labels.shape)

            # Forward pass of the student model
            student_logits, preds = final_student(student_input)
            # print("student_logits: ", student_logits.shape)  # not probability
            # print("preds: ", preds.shape)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = F.softmax(pseudo_logits / T, dim=1)
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

            loss.backward()

            optimizer.step()
            running_loss += loss * batch_size
            running_kl += kl_loss_weight * kl_div_res * batch_size
            running_dice += ce_loss_weight * ce_res * batch_size
            running_focal += dice_loss_weight * dice_res * batch_size

        # Loop for the second data loader
        for i, (images, _) in enumerate(dl_stud2):
            print("# epoch: ", epoch, " - i: ", i)
            batch_size = images.shape[0]
            n += batch_size

            student_input = student_inputs(images, device, processor)

            optimizer.zero_grad()

            # Forward pass of the teacher model - do not save gradients to not change the teacher's weights
            with torch.no_grad():
                pseudo_logits, pseudo_labels = student2(student_input)
            # print("teacher_logits: ", teacher_logits.shape)
            # print("pseudo_labels: ", pseudo_labels.shape)

            # Forward pass of the student model
            student_logits, preds = final_student(student_input)
            # print("student_logits: ", student_logits.shape)  # not probability
            # print("preds: ", preds.shape)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = F.softmax(pseudo_logits / T, dim=1)
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

            loss.backward()

            optimizer.step()
            running_loss += loss * batch_size
            running_kl += kl_loss_weight * kl_div_res * batch_size
            running_dice += ce_loss_weight * ce_res * batch_size
            running_focal += dice_loss_weight * dice_res * batch_size

        train_loss.append(running_loss.detach().cpu() / n)
        kl_l.append(running_kl.detach().cpu() / n)
        dice_l.append(running_dice.detach().cpu() / n)
        focal_l.append(running_focal.detach().cpu() / n)

        # TODO validation (quale/i data loader e quale/i student?)

        # Save and plot
        if (epoch + 1) % 1 == 0:  # todo %10

            if path_to_save_model is not None:
                checkpoint_path = path_to_save_model + f'student_ckpt_epoch_{epoch + 1}.pth'
                torch.save(final_student.state_dict(), checkpoint_path)

            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), train_loss, label='Training Loss', marker='o')
            plt.plot(range(epoch + 1), val_loss, label='Validation Loss', marker='o')

            plt.plot(range(epoch + 1), kl_l, label='KL Loss', marker='o')
            plt.plot(range(epoch + 1), dice_l, label='Dice Loss', marker='o')
            plt.plot(range(epoch + 1), focal_l, label='Focal Loss', marker='o')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Losses')
            plt.legend()
            plt.grid(True)

            # Save the loss plot
            '''loss_plot_path = path_to_save_model + f'loss_plot_epoch_{epoch + 1}.png'
            plt.savefig(loss_plot_path)'''

            plt.show()

            visualize_segmentation(pseudo_labels[0])
            visualize_segmentation(pseudo_labels[1])

            visualize_segmentation(preds[0])
            visualize_segmentation(preds[1])


path_to_save_model_root = "./checkpoints/"
INTERMEDIATE_TRAINING = False


if __name__ == '__main__':
    print("### Starting Multistage Loop ###")
    if INTERMEDIATE_TRAINING:
        t0 = time.time()
        train(stud_id=1, path_to_save_model=path_to_save_model_root + "student1/")
        t1 = time.time()
        train(stud_id=2, path_to_save_model=path_to_save_model_root + "student2/")
        print("training/validation time for the intermediate students: {0:.2f}s".format(t1 - t0))

    t2 = time.time()
    final_train(path_to_save_model_root + "final_student")
    t3 = time.time()
    print("training/validation time for the final student: {0:.2f}s".format(t3 - t2))
    print("### Multistage Loop Terminated ###")
