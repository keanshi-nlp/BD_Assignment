"""模型工厂 - 使用torchvision预训练模型"""

import torch.nn as nn
from torchvision import models
import config


def get_model(model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False):
    """
    获取模型
    
    Args:
        model_name: 模型名称
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    """
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 适配CIFAR-100的32x32输入（修改第一个卷积层）
    if 'resnet' in model_name:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # 移除maxpool，因为CIFAR图像较小
    
    return model


def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable