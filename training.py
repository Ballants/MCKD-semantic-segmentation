import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from transformers import OneFormerProcessor, AutoModelForUniversalSegmentation

from model.encoder_decoder import SwinDeepLabV3Plus
from utils.losses import DiceLoss
from utils.visualization import visualize_segmentation


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

    # TODO l'ho aggiunto io. Il post-processor fa l'argmax direttamente sui logits
    semantic_segmentation = segmentation_logits.softmax(dim=1)

    semantic_segmentation = semantic_segmentation.argmax(dim=1)

    return segmentation_logits, semantic_segmentation


def train(stud_id, path_to_save_model=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()

    processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large").to(device)

    # processor_student = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224") # inutile
    student = SwinDeepLabV3Plus(num_classes=133).to(device)

    # freezing backbone's parameters
    for param in student.backbone.parameters():
        param.requires_grad = False

    # Load dataloaders
    train_dl = torch.load(f'./data_loaders/Train_dl_{stud_id}.pt')
    val_dl = torch.load(f'./data_loaders/Validation_dl_{stud_id}.pt')

    # Hyper-params settings
    learning_rate = 1e-03
    epochs = 100
    T = 10  # Todo

    # Weights sum up to 1
    # TODO paper 2015 dice che deve essere più alto al resto... online è sempre più basso del resto (tipo regularization)
    kl_loss_weight = 0.2
    ce_loss_weight = 0.4
    dice_loss_weight = 0.4

    # Todo capire se possono essere utili
    clip_grad = None
    use_scheduler = True
    '''if use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) '''

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)  # todo use adam?

    # Todo training loop
    teacher.eval()  # Teacher set to evaluation mode

    train_loss = []
    val_loss = []

    # TODO add running loss and fix training loop
    for epoch in range(epochs):

        student.train()
        running_loss = 0
        n = 0
        for i, (images, _) in enumerate(train_dl):
            print("# epoch: ", epoch, " - i: ", i)
            batch_size = images.shape[0]
            n += batch_size

            semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                do_rescale=False).to(device)
            semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)
            # print("pixel_values: ", semantic_inputs['pixel_values'].shape)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)
            # print("teacher_logits: ", teacher_logits.shape)
            # print("pseudo_labels: ", pseudo_labels.shape)

            # Forward pass with the student model
            student_logits, preds = student(semantic_inputs["pixel_values"])
            # print("student_logits: ", student_logits.shape)  # not probability
            # print("preds: ", preds.shape)

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

            '''print("####### epoch: ", epoch, " #######")
            print("KL Loss: ", soft_targets_loss)
            print("CE Loss: ", label_loss)
            print("Loss: ", loss)
            print("\n")'''

            loss.backward()

            # Todo togliere?
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)

            optimizer.step()
            running_loss += loss * batch_size

        train_loss.append(running_loss.detach().cpu() / n)

        # Todo validation: compute metrics
        student.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0
            for i, (images, _) in enumerate(val_dl):
                batch_size = images.shape[0]
                n += batch_size

                semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                    do_rescale=False).to(device)
                semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)
                # print("pixel_values: ", semantic_inputs['pixel_values'].shape)

                teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)

                student_logits, preds = student(semantic_inputs["pixel_values"])

                soft_targets = F.softmax(teacher_logits / T, dim=1)
                soft_prob = F.log_softmax(student_logits / T, dim=1)

                # [batch * width * height, classes]
                soft_targets = soft_targets.permute(0, 2, 3, 1).reshape(-1, 133)
                soft_prob = soft_prob.permute(0, 2, 3, 1).reshape(-1, 133)

                kl_div_res = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)
                ce_res = ce_loss(student_logits, pseudo_labels)
                dice_res = dice_loss(student_logits, pseudo_labels)

                loss = kl_loss_weight * kl_div_res + ce_loss_weight * ce_res + dice_loss_weight * dice_res

                '''print("####### epoch: ", epoch, " #######")
                print("KL Loss: ", soft_targets_loss)
                print("CE Loss: ", label_loss)
                print("Loss: ", loss)
                print("\n")'''

                running_loss += loss * batch_size

            val_loss.append(running_loss.detach().cpu() / n)

        '''if use_scheduler:
            scheduler.step()'''

        # Save and plot
        if (epoch + 1) % 1 == 0:  # todo %10

            if path_to_save_model is not None:
                checkpoint_path = path_to_save_model + f'student_ckpt_epoch_{epoch + 1}.pth'
                torch.save(student.state_dict(), checkpoint_path)

            plt.figure(figsize=(10, 6))
            plt.plot(range(epoch + 1), train_loss, label='Training Loss', marker='o')
            plt.plot(range(epoch + 1), val_loss, label='Validation Loss', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
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


if __name__ == '__main__':
    print("### Starting training... ###")
    t0 = time.time()
    train(stud_id=1)
    t1 = time.time()
    print("training/validation time: {0:.2f}s".format(t1 - t0))
    print("### DataLoader ready ###")
