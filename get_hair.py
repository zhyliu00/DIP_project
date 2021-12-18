import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from io import StringIO
import matplotlib.pyplot as plt
import cv2
# torch vision lib
# from engine import train_one_epoch, evaluate
# import utils
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def getPic(img_rgb, transforms=None):
    # img_path = f"{data_dir}/{img_name}"
    # img = Image.open(img_path).convert("RGB")
    img = img_rgb
    target = {}
    target["image_name"] = 'input'
    if transforms is not None:
        img, target = transforms(img, target)
    return img, target

      
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_hair_mask(img_rgb):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    model_path = "./preTrainModel/final20_complete_epoch.pth"
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    temp_transforms = get_transform(train=False)
    img, img_info = getPic(img_rgb, temp_transforms)
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    mask = prediction[0]['masks'][:, 0][0].detach().cpu().numpy()
    # real_img = cv2.imread(f"{data_dir}/{img_info['image_name']}")
    real_img = img_rgb
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    real_img[mask<0.5]=255 # 0.5是阈值 可灵活调整
    return real_img
        
def get_hair_rgb(img_rgb):
    # 图像存储路径
    # data_dir = "./dataset/FRLL-Morphs/neutral_front"
    # 图像名
    # img_name = "001_03.jpg"
    masked_img = get_hair_mask(img_rgb)

	# 存储获取的mask图像查看效果
    cv2.imwrite("./just.jpg",masked_img)

    return masked_img

