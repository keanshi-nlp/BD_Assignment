"""
实验5: 混合并行 (数据并行 + 模型并行/张量并行)
使用FSDP (Fully Sharded Data Parallel) 实现更高效的混合并行
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import argparse

import config
from data_loader import get_dataloaders
from model_factory import get_model, count_parameters
from utils import (train_epoch, evaluate, get_gpu_memory, 
                   print_training_info, MetricsLogger)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train_fsdp(rank, world_size, args):
    """FSDP训练"""
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"FSDP Training with {world_size} GPUs")
        print(f"Sharding Strategy: {args.sharding}")
    
    # 加载数据
    train_loader, test_loader, train_sampler = get_dataloaders(
        batch_size=args.batch_size,
        distributed=True,
        rank=rank,
        world_size=world_size,
        use_cache=args.use_cache
    )
    
    # 创建模型
    model = get_model(args.model)
    
    # FSDP配置
    # 混合精度配置
    mp_policy = None
    if args.amp:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # 分片策略
    if args.sharding == 'full':
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args.sharding == 'grad':
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding == 'no':
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # 自动包装策略
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1e6
    )
    
    # 包装模型
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=rank,
    )
    
    if rank == 0:
        print(f"Model wrapped with FSDP")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr * world_size, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 日志
    if rank == 0:
        exp_name = f"fsdp_{args.sharding}_{args.model}_bs{args.batch_size}x{world_size}_amp{args.amp}"
        logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.reset_peak_memory_stats()
        
        train_loss, train_acc, epoch_time, throughput = train_epoch(
            model, train_loader, criterion, optimizer, device, None, False
        )
        
        # 同步指标
        metrics = torch.tensor([train_loss, train_acc, throughput], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        train_loss, train_acc = metrics[0].item(), metrics[1].item()
        total_throughput = throughput * world_size
        
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
    
    if rank == 0:
        extra_info = {
            'model': args.model,
            'batch_size_per_gpu': args.batch_size,
            'effective_batch_size': args.batch_size * world_size,
            'world_size': world_size,
            'sharding_strategy': args.sharding,
            'use_amp': args.amp,
            'best_acc': best_acc,
            'training_type': 'fsdp_hybrid_parallel'
        }
        logger.save(extra_info)
        print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='FSDP Hybrid Parallel Training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--sharding', type=str, default='full', 
                        choices=['full', 'grad', 'no'],
                        help='FSDP sharding strategy')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--amp', action='store_true')
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    torch.multiprocessing.spawn(
        train_fsdp,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()