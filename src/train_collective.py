"""
实验: Collective 模式 (AllReduce)
使用 DDP 实现，作为与 Parameter Server 的对比基线
支持 NVTX 性能分析

通信模式: Ring-AllReduce（NCCL 后端默认）
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
from utils import (train_epoch_ddp, evaluate, get_gpu_memory, 
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


def train_collective(rank, world_size, args):
    """Collective 模式训练（AllReduce）"""
    setup(rank, world_size, args.backend)
    
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"Collective Mode Training ({args.backend.upper()} AllReduce)")
        print(f"World size: {world_size}, Backend: {args.backend}")
        if args.profile:
            print("NVTX profiling enabled")
    
    # 加载数据
    train_loader, test_loader, train_sampler = get_dataloaders(
        batch_size=args.batch_size,
        distributed=True,
        rank=rank,
        world_size=world_size,
        use_cache=args.use_cache
    )
    
    if rank == 0:
        print(f"Per-GPU batch size: {args.batch_size}")
        print(f"Global batch size: {args.batch_size * world_size}")
    
    # 创建模型
    model = get_model(args.model).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        total_params, _ = count_parameters(model.module)
        print(f"Model: {args.model}, Parameters: {total_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr,
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 混合精度
    scaler = GradScaler() if args.amp else None
    
    # 日志
    if rank == 0:
        exp_name = f"collective_{args.backend}_{args.model}_bs{args.batch_size}x{world_size}"
        logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        
        # 使用带NVTX标记的DDP训练函数
        train_loss, train_acc, epoch_time, throughput = train_epoch_ddp(
            model, train_loader, criterion, optimizer, device, 
            scaler, args.amp, profile=args.profile
        )
        
        # AllReduce 同步指标
        metrics = torch.tensor([train_loss, train_acc, throughput], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        train_loss, train_acc = metrics[0].item(), metrics[1].item()
        total_throughput = throughput * world_size
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                        profile=args.profile)
        
        scheduler.step()
        gpu_memory = get_gpu_memory()
        
        if rank == 0:
            print_training_info(epoch, args.epochs, train_loss, train_acc, 
                                test_loss, test_acc, epoch_time, total_throughput, gpu_memory)
            logger.log(epoch, train_loss, train_acc, test_loss, test_acc, 
                       epoch_time, total_throughput, gpu_memory)
            
            if test_acc > best_acc:
                best_acc = test_acc
    
    if rank == 0:
        extra_info = {
            'model': args.model,
            'batch_size_per_gpu': args.batch_size,
            'effective_batch_size': args.batch_size * world_size,
            'world_size': world_size,
            'backend': args.backend,
            'use_amp': args.amp,
            'best_acc': best_acc,
            'training_type': 'collective_allreduce',
            'communication_mode': 'collective'
        }
        logger.save(extra_info)
        print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Collective Mode Training (AllReduce)')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--profile', action='store_true', help='Enable NVTX profiling')
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    torch.multiprocessing.spawn(
        train_collective,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()