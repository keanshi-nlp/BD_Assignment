"""
实验2: 单机多卡 - 数据并行 (DDP)
使用PyTorch DistributedDataParallel
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import argparse

import config
from data_loader import get_dataloaders
from model_factory import get_model, count_parameters
from utils import (train_epoch, evaluate, get_gpu_memory, 
                   print_training_info, MetricsLogger)


def setup(rank, world_size, backend='nccl'):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train_ddp(rank, world_size, args):
    """DDP训练主函数"""
    setup(rank, world_size, args.backend)
    
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"DDP Training with {world_size} GPUs, Backend: {args.backend}")
    
    # 加载数据 - 分布式采样器
    # 注意: DDP下每个GPU的有效batch_size = batch_size
    # 总batch_size = batch_size * world_size
    train_loader, test_loader, train_sampler = get_dataloaders(
        batch_size=args.batch_size,
        distributed=True,
        rank=rank,
        world_size=world_size,
        use_cache=args.use_cache
    )
    
    if rank == 0:
        print(f"Per-GPU batch size: {args.batch_size}, Effective batch size: {args.batch_size * world_size}")
    
    # 创建模型并包装DDP
    model = get_model(args.model).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        total_params, trainable_params = count_parameters(model.module)
        print(f"Model: {args.model}, Parameters: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 学习率按world_size线性缩放
    scaled_lr = args.lr * world_size if args.scale_lr else args.lr
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度
    scaler = GradScaler() if args.amp else None
    
    # 日志
    if rank == 0:
        exp_name = f"ddp_{args.backend}_{args.model}_bs{args.batch_size}x{world_size}_cache{args.use_cache}_amp{args.amp}"
        logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # 确保每个epoch的数据打乱不同
        torch.cuda.reset_peak_memory_stats()
        
        train_loss, train_acc, epoch_time, throughput = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.amp
        )
        
        # 汇总所有GPU的指标
        metrics = torch.tensor([train_loss, train_acc, throughput], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        train_loss, train_acc = metrics[0].item(), metrics[1].item()
        total_throughput = throughput * world_size  # 总吞吐量
        
        # 评估 (只在rank 0上进行完整评估)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        gpu_memory = get_gpu_memory()
        
        if rank == 0:
            print_training_info(epoch, args.epochs, train_loss, train_acc, 
                                test_loss, test_acc, epoch_time, total_throughput, gpu_memory)
            logger.log(epoch, train_loss, train_acc, test_loss, test_acc, 
                       epoch_time, total_throughput, gpu_memory)
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.module.state_dict(), 
                           f"{config.CHECKPOINT_DIR}/{exp_name}_best.pth")
    
    if rank == 0:
        extra_info = {
            'model': args.model,
            'batch_size_per_gpu': args.batch_size,
            'effective_batch_size': args.batch_size * world_size,
            'world_size': world_size,
            'backend': args.backend,
            'use_cache': args.use_cache,
            'use_amp': args.amp,
            'scaled_lr': scaled_lr,
            'best_acc': best_acc,
            'training_type': 'ddp_data_parallel'
        }
        logger.save(extra_info)
        print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='DDP Data Parallel Training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--scale_lr', action='store_true', help='Scale LR by world size')
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()