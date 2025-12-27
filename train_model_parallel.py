"""
实验: 模型并行 (Pipeline Parallelism)
将模型分割到多个GPU上
支持 NVTX 性能分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

import config
from data_loader import get_dataloaders
from model_factory import count_parameters
from utils import (train_epoch_model_parallel, evaluate_model_parallel, 
                   get_gpu_memory, print_training_info, MetricsLogger)


class ModelParallelResNet(nn.Module):
    """
    模型并行版ResNet
    将模型分割到两个GPU上
    """
    def __init__(self, base_model, device0, device1):
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        
        # 第一部分在GPU0: conv1 -> layer2
        self.part1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
        ).to(device0)
        
        # 第二部分在GPU1: layer3 -> fc
        self.part2 = nn.Sequential(
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        ).to(device1)
        
        self.fc = base_model.fc.to(device1)
    
    def forward(self, x):
        # 第一部分在GPU0
        x = self.part1(x)
        # 传输到GPU1
        x = x.to(self.device1)
        # 第二部分在GPU1
        x = self.part2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ModelParallelVGG(nn.Module):
    """
    模型并行版VGG
    """
    def __init__(self, base_model, device0, device1):
        super().__init__()
        self.device0 = device0
        self.device1 = device1
        
        # 分割features
        features = list(base_model.features.children())
        mid = len(features) // 2
        
        self.features1 = nn.Sequential(*features[:mid]).to(device0)
        self.features2 = nn.Sequential(*features[mid:]).to(device1)
        self.avgpool = base_model.avgpool.to(device1)
        self.classifier = base_model.classifier.to(device1)
    
    def forward(self, x):
        x = self.features1(x)
        x = x.to(self.device1)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model_parallel(model_name, device0, device1):
    """获取模型并行版本的模型"""
    from model_factory import get_model
    base_model = get_model(model_name)
    
    if 'resnet' in model_name:
        return ModelParallelResNet(base_model, device0, device1)
    elif 'vgg' in model_name:
        return ModelParallelVGG(base_model, device0, device1)
    else:
        raise ValueError(f"Model parallel not supported for {model_name}")


def train_model_parallel(args):
    """模型并行训练"""
    # 设置设备
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    print(f"Model Parallel Training: {args.model}")
    print(f"Device 0: {device0}, Device 1: {device1}")
    if args.profile:
        print("NVTX profiling enabled")
    
    # 加载数据
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        distributed=False,
        use_cache=args.use_cache
    )
    print(f"Batch size: {args.batch_size}")
    
    # 创建模型并行模型
    model = get_model_parallel(args.model, device0, device1)
    total_params, _ = count_parameters(model)
    print(f"Parameters: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 日志
    exp_name = f"model_parallel_{args.model}_bs{args.batch_size}_cache{args.use_cache}"
    logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        torch.cuda.reset_peak_memory_stats()
        
        # 使用带NVTX标记的模型并行训练函数
        train_loss, train_acc, epoch_time, throughput = train_epoch_model_parallel(
            model, train_loader, criterion, optimizer, 
            device0, device1, profile=args.profile
        )
        
        test_loss, test_acc = evaluate_model_parallel(
            model, test_loader, criterion, device0, device1, profile=args.profile
        )
        
        scheduler.step()
        gpu_memory = get_gpu_memory()
        
        print_training_info(epoch, args.epochs, train_loss, train_acc, 
                            test_loss, test_acc, epoch_time, throughput, gpu_memory)
        
        logger.log(epoch, train_loss, train_acc, test_loss, test_acc, 
                   epoch_time, throughput, gpu_memory)
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    # 保存结果
    extra_info = {
        'model': args.model,
        'batch_size': args.batch_size,
        'use_cache': args.use_cache,
        'best_acc': best_acc,
        'training_type': 'model_parallel'
    }
    logger.save(extra_info)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Model Parallel Training')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'vgg16'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Enable NVTX profiling')
    
    args = parser.parse_args()
    
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Model parallel requires at least 2 GPUs")
    
    train_model_parallel(args)


if __name__ == '__main__':
    main()