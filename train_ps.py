"""
实验6: Parameter Server 模式
使用PyTorch RPC实现参数服务器架构
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import time
import argparse

import config
from data_loader import get_dataloaders
from model_factory import get_model
from utils import AverageMeter, MetricsLogger


# 参数服务器类
class ParameterServer:
    def __init__(self, model_name):
        self.model = get_model(model_name)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                   momentum=0.9, weight_decay=5e-4)
        
    def get_model(self):
        return self.model
    
    def get_param_rrefs(self):
        return [RRef(p) for p in self.model.parameters()]
    
    @staticmethod
    def average_gradients(param_server_rref):
        """平均梯度"""
        pass


# Worker类
class Worker:
    def __init__(self, ps_rref, rank):
        self.ps_rref = ps_rref
        self.rank = rank
        self.criterion = nn.CrossEntropyLoss()
        
    def train_batch(self, data, target):
        """训练一个batch"""
        with dist_autograd.context() as context_id:
            model = self.ps_rref.rpc_sync().get_model()
            output = model(data)
            loss = self.criterion(output, target)
            dist_autograd.backward(context_id, [loss])
            return loss.item()


def run_parameter_server(args):
    """
    简化版Parameter Server实现
    使用torch.distributed模拟PS架构
    """
    import torch.distributed as dist
    
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    
    # 初始化进程组
    dist.init_process_group(backend='gloo', rank=0, world_size=2)
    
    device = torch.device('cuda:0')
    
    print("Parameter Server Mode (Simulated)")
    print("Note: PyTorch's native PS mode has limited support, using gradient accumulation simulation")
    
    # 加载数据
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        distributed=False,
        use_cache=args.use_cache
    )
    
    # 创建模型
    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 日志
    exp_name = f"ps_mode_{args.model}_bs{args.batch_size}"
    logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 模拟PS训练：使用梯度累积模拟多worker
    num_workers = 2
    accumulation_steps = num_workers
    
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        losses = AverageMeter()
        accs = AverageMeter()
        
        start_time = time.time()
        total_samples = 0
        
        optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item() * accumulation_steps, data.size(0))
            accs.update(correct / data.size(0), data.size(0))
            total_samples += data.size(0)
        
        epoch_time = time.time() - start_time
        throughput = total_samples / epoch_time
        
        # 评估
        model.eval()
        test_losses = AverageMeter()
        test_accs = AverageMeter()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                test_losses.update(loss.item(), data.size(0))
                test_accs.update(correct / data.size(0), data.size(0))
        
        scheduler.step()
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        train_acc = accs.avg * 100
        test_acc = test_accs.avg * 100
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {losses.avg:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_losses.avg:.4f} Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s | Throughput: {throughput:.1f} samples/s")
        
        logger.log(epoch, losses.avg, train_acc, test_losses.avg, test_acc,
                   epoch_time, throughput, gpu_memory)
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    extra_info = {
        'model': args.model,
        'batch_size': args.batch_size,
        'num_workers_simulated': num_workers,
        'best_acc': best_acc,
        'training_type': 'parameter_server_simulated'
    }
    logger.save(extra_info)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Parameter Server Training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--use_cache', action='store_true')
    
    args = parser.parse_args()
    run_parameter_server(args)


if __name__ == '__main__':
    main()