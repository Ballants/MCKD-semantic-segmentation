import torch

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, AutoModelForUniversalSegmentation, SwinForImageClassification, AutoImageProcessor
from encoder_decoder import SwinDeepLabV3Plus

def train(train_loader, stud_id):
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

    # Todo training loop

def main():
    pass


