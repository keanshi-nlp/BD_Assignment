"""训练工具函数"""

import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import json
import os
from datetime import datetime


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsLogger:
    """指标记录器"""
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': [],
            'throughput': [],  # samples/sec
            'gpu_memory': [],
        }
        self.start_time = time.time()
        
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc, epoch_time, throughput, gpu_memory):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['epoch_time'].append(epoch_time)
        self.metrics['throughput'].append(throughput)
        self.metrics['gpu_memory'].append(gpu_memory)
        
    def save(self, extra_info=None):
        result = {
            'experiment_name': self.experiment_name,
            'total_time': time.time() - self.start_time,
            'metrics': self.metrics,
        }
        if extra_info:
            result.update(extra_info)
            
        filename = os.path.join(self.log_dir, f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Metrics saved to {filename}")
        return filename


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_amp=False):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        losses.update(loss.item(), data.size(0))
        accs.update(correct / data.size(0), data.size(0))
        total_samples += data.size(0)
    
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time
    
    return losses.avg, accs.avg * 100, epoch_time, throughput


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item(), data.size(0))
            accs.update(correct / data.size(0), data.size(0))
    
    return losses.avg, accs.avg * 100


def get_gpu_memory():
    """获取GPU内存使用情况(MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def print_training_info(epoch, epochs, train_loss, train_acc, test_loss, test_acc, 
                        epoch_time, throughput, gpu_memory, rank=0):
    """打印训练信息"""
    if rank == 0:
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s | "
              f"Throughput: {throughput:.1f} samples/s | "
              f"GPU Mem: {gpu_memory:.0f}MB")