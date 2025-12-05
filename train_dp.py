"""
实验3: 单机多卡 - DataParallel (DP)
使用PyTorch DataParallel (作为对比基线)
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


def train_dp(args):
    """DataParallel训练"""
    # 检测可用GPU
    device_ids = list(range(torch.cuda.device_count()))
    print(f"DataParallel Training with GPUs: {device_ids}")
    
    device = torch.device('cuda:0')
    
    # 加载数据
    # DP模式下，batch_size是总的batch size
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        distributed=False,
        use_cache=args.use_cache
    )
    
    # 创建模型并包装DataParallel
    model = get_model(args.model).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    
    total_params, trainable_params = count_parameters(model.module)
    print(f"Model: {args.model}, Parameters: {total_params:,}")
    print(f"Total batch size: {args.batch_size}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度
    scaler = GradScaler() if args.amp else None
    
    # 日志
    exp_name = f"dp_{args.model}_bs{args.batch_size}_ngpu{len(device_ids)}_cache{args.use_cache}_amp{args.amp}"
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
            torch.save(model.module.state_dict(), 
                       f"{config.CHECKPOINT_DIR}/{exp_name}_best.pth")
    
    extra_info = {
        'model': args.model,
        'batch_size': args.batch_size,
        'num_gpus': len(device_ids),
        'use_cache': args.use_cache,
        'use_amp': args.amp,
        'best_acc': best_acc,
        'training_type': 'data_parallel'
    }
    logger.save(extra_info)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataParallel Training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=256, help='Total batch size')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--amp', action='store_true')
    
    args = parser.parse_args()
    train_dp(args)