"""
实验1: 单机单卡训练 (非分布式基线)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import argparse

import config
from data_loader import get_dataloaders
from model_factory import get_model, count_parameters
from utils import (train_epoch, evaluate, get_gpu_memory, 
                   print_training_info, MetricsLogger)


def train_single_gpu(args):
    """单卡训练"""
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        distributed=False,
        use_cache=args.use_cache
    )
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # 创建模型
    model = get_model(args.model).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {args.model}, Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练
    scaler = GradScaler() if args.amp else None
    
    # 日志记录
    exp_name = f"single_gpu_{args.model}_bs{args.batch_size}_cache{args.use_cache}_amp{args.amp}"
    logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        torch.cuda.reset_peak_memory_stats()
        
        train_loss, train_acc, epoch_time, throughput = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.amp
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        gpu_memory = get_gpu_memory()
        
        print_training_info(epoch, args.epochs, train_loss, train_acc, 
                            test_loss, test_acc, epoch_time, throughput, gpu_memory)
        
        logger.log(epoch, train_loss, train_acc, test_loss, test_acc, 
                   epoch_time, throughput, gpu_memory)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 
                       f"{config.CHECKPOINT_DIR}/{exp_name}_best.pth")
    
    # 保存结果
    extra_info = {
        'model': args.model,
        'batch_size': args.batch_size,
        'use_cache': args.use_cache,
        'use_amp': args.amp,
        'best_acc': best_acc,
        'total_params': total_params,
        'gpu': args.gpu,
        'training_type': 'single_gpu'
    }
    logger.save(extra_info)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single GPU Training')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                                 'vgg16', 'densenet121', 'efficientnet_b0'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_cache', action='store_true', help='Cache data in memory')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    train_single_gpu(args)