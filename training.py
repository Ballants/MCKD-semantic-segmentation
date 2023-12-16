import torch
import torch.nn as nn
import torch.optim as optim
from transformers import OneFormerProcessor, AutoModelForUniversalSegmentation

from model.encoder_decoder import SwinDeepLabV3Plus
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
    print("sum: ", torch.sum(segmentation_logits[:, :, 0, 0], dim=1))

    # TODO l'ho aggiunto io. Il post-processor fa l'argmax direttamente sui logits
    semantic_segmentation = segmentation_logits.softmax(dim=1)

    semantic_segmentation = semantic_segmentation.argmax(dim=1)

    return segmentation_logits, semantic_segmentation


def train(stud_id):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())
    print('Device: ', device)
    torch.cuda.empty_cache()

    processor_teacher = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    teacher = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    # processor_student = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224") # inutile
    student = SwinDeepLabV3Plus(num_classes=133)

    # freezing backbone's parameters
    for param in student.backbone.parameters():
        param.requires_grad = False

    # Load dataloaders
    train_dl = torch.load(f'./data_loaders/Train_dl_{stud_id}.pt')
    val_dl = torch.load(f'./data_loaders/Validation_dl_{stud_id}.pt')

    # Todo to.device()

    # Hyper-params settings
    learning_rate = 1e-04
    epochs = 10
    T = 10
    soft_target_loss_weight = 0.25
    ce_loss_weight = 1 - soft_target_loss_weight

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    # Todo training loop
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    # TODO add running loss and fix training loop
    for e in range(epochs):
        for i, (images, targets) in enumerate(train_dl):
            batch_size = images.shape[0]
            images = images.to(device)
            semantic_inputs = processor_teacher(images=images, task_inputs=["semantic"], return_tensors="pt",
                                                do_rescale=False)
            semantic_inputs["task_inputs"] = semantic_inputs["task_inputs"].repeat(batch_size, 1)
            print("pixel_values: ", semantic_inputs['pixel_values'].shape)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits, pseudo_labels = teacher_forward(teacher, **semantic_inputs)
            print("teacher_logits: ", teacher_logits.shape)
            print("pseudo_labels: ", pseudo_labels.shape)

            # Forward pass with the student model
            student_logits, preds = student(semantic_inputs["pixel_values"])
            print("student_logits: ", student_logits.shape)  # not probability
            print("preds: ", preds.shape)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. ["Distilling the knowledge in a neural network"]
            # KL Div:
            soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T ** 2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, pseudo_labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
            print("####### epoch: ", e, " #######")
            print("KL Loss: ", soft_targets_loss)
            print("CE Loss: ", label_loss)
            print("Loss: ", loss)
            print("\n")

            loss.backward()
            optimizer.step()

    visualize_segmentation(pseudo_labels[0])
    visualize_segmentation(pseudo_labels[1])

    visualize_segmentation(preds[0])
    visualize_segmentation(preds[1])


if __name__ == '__main__':
    print("### Starting training... ###")
    train(stud_id=1)
    print("### DataLoader ready ###")
