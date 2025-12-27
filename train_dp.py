"""
实验2: DataParallel (DP) 训练
支持 NVTX 性能分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import argparse

import config
from data_loader import get_dataloaders
from model_factory import get_model, count_parameters
from utils import (train_epoch_dp, evaluate, get_gpu_memory, 
                   print_training_info, MetricsLogger)


def train_dp(args):
    """DataParallel 训练"""
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    if args.profile:
        print("NVTX profiling enabled")
    
    # 主设备
    device = torch.device('cuda:0')
    
    # 加载数据
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,  # DP: 总batch size
        distributed=False,
        use_cache=args.use_cache
    )
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    print(f"Total batch size: {args.batch_size} (split across {num_gpus} GPUs)")
    
    # 创建模型
    model = get_model(args.model)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {args.model}, Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    # 使用 DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度训练
    scaler = GradScaler() if args.amp else None
    
    # 日志记录
    exp_name = f"dp_{args.model}_bs{args.batch_size}_ngpu{num_gpus}_cache{args.use_cache}_amp{args.amp}"
    logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        torch.cuda.reset_peak_memory_stats()
        
        train_loss, train_acc, epoch_time, throughput = train_epoch_dp(
            model, train_loader, criterion, optimizer, device, 
            scaler, args.amp, profile=args.profile
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                        profile=args.profile)
        
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
        'num_gpus': num_gpus,
        'use_cache': args.use_cache,
        'use_amp': args.amp,
        'best_acc': best_acc,
        'total_params': total_params,
        'training_type': 'data_parallel'
    }
    logger.save(extra_info)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataParallel Training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=256, help='Total batch size')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Enable NVTX profiling')
    
    args = parser.parse_args()
    train_dp(args)