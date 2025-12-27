"""工具函数模块 - 包含NVTX性能分析支持"""

import os
import time
import json
import torch
import torch.cuda.nvtx as nvtx
from contextlib import contextmanager

import config


# ============================================
# NVTX 性能分析工具
# ============================================

class NVTXRange:
    """NVTX 范围标记器（上下文管理器）"""
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()
    
    def __enter__(self):
        if self.enabled:
            nvtx.range_push(self.name)
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            nvtx.range_pop()


@contextmanager
def nvtx_range(name, enabled=True):
    """NVTX 范围标记器（函数式）"""
    if enabled and torch.cuda.is_available():
        nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled and torch.cuda.is_available():
            nvtx.range_pop()


def nvtx_mark(name, enabled=True):
    """NVTX 单点标记"""
    if enabled and torch.cuda.is_available():
        nvtx.mark(name)


# ============================================
# 训练工具类
# ============================================

class AverageMeter:
    """计算和存储平均值"""
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
    """实验指标记录器"""
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': [],
            'throughput': [],
            'gpu_memory': [],
        }
        self.start_time = time.time()
    
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc, 
            epoch_time, throughput, gpu_memory):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['epoch_time'].append(epoch_time)
        self.metrics['throughput'].append(throughput)
        self.metrics['gpu_memory'].append(gpu_memory)
    
    def save(self, extra_info=None):
        total_time = time.time() - self.start_time
        result = {
            'experiment_name': self.experiment_name,
            'metrics': self.metrics,
            'total_time': total_time,
        }
        if extra_info:
            result.update(extra_info)
        
        os.makedirs(self.log_dir, exist_ok=True)
        filepath = os.path.join(self.log_dir, f"{self.experiment_name}.json")
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {filepath}")


# ============================================
# 训练和评估函数（带NVTX标记）
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device, 
                scaler=None, use_amp=False, profile=False):
    """
    训练一个epoch（带NVTX性能标记）
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        scaler: AMP的GradScaler
        use_amp: 是否使用混合精度
        profile: 是否启用NVTX标记
    """
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        with NVTXRange("Batch", enabled=profile):
            
            # 数据传输到GPU
            with NVTXRange("DataToDevice", enabled=profile):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # 梯度清零
            with NVTXRange("ZeroGrad", enabled=profile):
                optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    with NVTXRange("Forward_AMP", enabled=profile):
                        output = model(data)
                    
                    with NVTXRange("Loss_AMP", enabled=profile):
                        loss = criterion(output, target)
                
                with NVTXRange("Backward_AMP", enabled=profile):
                    scaler.scale(loss).backward()
                
                with NVTXRange("OptimizerStep_AMP", enabled=profile):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # 普通精度训练
                with NVTXRange("Forward", enabled=profile):
                    output = model(data)
                
                with NVTXRange("Loss", enabled=profile):
                    loss = criterion(output, target)
                
                with NVTXRange("Backward", enabled=profile):
                    loss.backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
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


def train_epoch_ddp(model, train_loader, criterion, optimizer, device,
                    scaler=None, use_amp=False, profile=False):
    """
    DDP训练一个epoch（带NVTX性能标记，标注AllReduce）
    """
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        with NVTXRange("Batch", enabled=profile):
            
            with NVTXRange("DataToDevice", enabled=profile):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with NVTXRange("ZeroGrad", enabled=profile):
                optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    with NVTXRange("Forward", enabled=profile):
                        output = model(data)
                    
                    with NVTXRange("Loss", enabled=profile):
                        loss = criterion(output, target)
                
                # DDP: backward包含隐式AllReduce
                with NVTXRange("Backward_AllReduce", enabled=profile):
                    scaler.scale(loss).backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                with NVTXRange("Forward", enabled=profile):
                    output = model(data)
                
                with NVTXRange("Loss", enabled=profile):
                    loss = criterion(output, target)
                
                # DDP: backward包含隐式AllReduce
                with NVTXRange("Backward_AllReduce", enabled=profile):
                    loss.backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
                    optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item(), data.size(0))
            accs.update(correct / data.size(0), data.size(0))
            total_samples += data.size(0)
    
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time
    
    return losses.avg, accs.avg * 100, epoch_time, throughput


def train_epoch_fsdp(model, train_loader, criterion, optimizer, device,
                     scaler=None, use_amp=False, profile=False):
    """
    FSDP训练一个epoch（带NVTX性能标记，标注AllGather/ReduceScatter）
    """
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        with NVTXRange("Batch", enabled=profile):
            
            with NVTXRange("DataToDevice", enabled=profile):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with NVTXRange("ZeroGrad", enabled=profile):
                optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    # FSDP: forward包含AllGather
                    with NVTXRange("Forward_FSDP_AllGather", enabled=profile):
                        output = model(data)
                    
                    with NVTXRange("Loss", enabled=profile):
                        loss = criterion(output, target)
                
                # FSDP: backward包含ReduceScatter
                with NVTXRange("Backward_FSDP_ReduceScatter", enabled=profile):
                    scaler.scale(loss).backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # FSDP: forward包含AllGather
                with NVTXRange("Forward_FSDP_AllGather", enabled=profile):
                    output = model(data)
                
                with NVTXRange("Loss", enabled=profile):
                    loss = criterion(output, target)
                
                # FSDP: backward包含ReduceScatter
                with NVTXRange("Backward_FSDP_ReduceScatter", enabled=profile):
                    loss.backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
                    optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item(), data.size(0))
            accs.update(correct / data.size(0), data.size(0))
            total_samples += data.size(0)
    
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time
    
    return losses.avg, accs.avg * 100, epoch_time, throughput


def train_epoch_model_parallel(model, train_loader, criterion, optimizer,
                               device0, device1, scaler=None, use_amp=False, profile=False):
    """
    模型并行训练一个epoch（带NVTX性能标记）
    """
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        with NVTXRange("Batch", enabled=profile):
            
            # 输入到第一个设备
            with NVTXRange("DataToDevice0", enabled=profile):
                data = data.to(device0, non_blocking=True)
            
            # 标签到最后一个设备（输出所在位置）
            with NVTXRange("TargetToDevice1", enabled=profile):
                target = target.to(device1, non_blocking=True)
            
            with NVTXRange("ZeroGrad", enabled=profile):
                optimizer.zero_grad(set_to_none=True)
            
            # 模型并行：数据流经不同GPU
            with NVTXRange("Forward_Pipeline", enabled=profile):
                output = model(data)
            
            with NVTXRange("Loss", enabled=profile):
                loss = criterion(output, target)
            
            # 梯度反向流过各GPU
            with NVTXRange("Backward_Pipeline", enabled=profile):
                loss.backward()
            
            with NVTXRange("OptimizerStep", enabled=profile):
                optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item(), data.size(0))
            accs.update(correct / data.size(0), data.size(0))
            total_samples += data.size(0)
    
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time
    
    return losses.avg, accs.avg * 100, epoch_time, throughput


def train_epoch_dp(model, train_loader, criterion, optimizer, device,
                   scaler=None, use_amp=False, profile=False):
    """
    DataParallel训练一个epoch（带NVTX性能标记）
    DP 的特点：数据自动分发到多卡，梯度自动汇总到主卡
    """
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        with NVTXRange("Batch", enabled=profile):
            
            # 数据传输到主GPU（DP会自动分发到其他GPU）
            with NVTXRange("DataToDevice_DP", enabled=profile):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with NVTXRange("ZeroGrad", enabled=profile):
                optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    # DP: forward 会自动将数据分发到多卡并行计算
                    with NVTXRange("Forward_DP_Scatter", enabled=profile):
                        output = model(data)
                    
                    with NVTXRange("Loss", enabled=profile):
                        loss = criterion(output, target)
                
                # DP: backward 会自动将梯度汇总到主卡
                with NVTXRange("Backward_DP_Gather", enabled=profile):
                    scaler.scale(loss).backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # DP: forward 会自动将数据分发到多卡并行计算
                with NVTXRange("Forward_DP_Scatter", enabled=profile):
                    output = model(data)
                
                with NVTXRange("Loss", enabled=profile):
                    loss = criterion(output, target)
                
                # DP: backward 会自动将梯度汇总到主卡
                with NVTXRange("Backward_DP_Gather", enabled=profile):
                    loss.backward()
                
                with NVTXRange("OptimizerStep", enabled=profile):
                    optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item(), data.size(0))
            accs.update(correct / data.size(0), data.size(0))
            total_samples += data.size(0)
    
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time
    
    return losses.avg, accs.avg * 100, epoch_time, throughput


def evaluate(model, test_loader, criterion, device, profile=False):
    """评估模型（带NVTX标记）"""
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    
    with NVTXRange("Evaluation", enabled=profile):
        with torch.no_grad():
            for data, target in test_loader:
                with NVTXRange("EvalBatch", enabled=profile):
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    
                    losses.update(loss.item(), data.size(0))
                    accs.update(correct / data.size(0), data.size(0))
    
    return losses.avg, accs.avg * 100


def evaluate_model_parallel(model, test_loader, criterion, device0, device1, profile=False):
    """评估模型并行模型"""
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    
    with NVTXRange("Evaluation", enabled=profile):
        with torch.no_grad():
            for data, target in test_loader:
                with NVTXRange("EvalBatch", enabled=profile):
                    data = data.to(device0, non_blocking=True)
                    target = target.to(device1, non_blocking=True)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    
                    losses.update(loss.item(), data.size(0))
                    accs.update(correct / data.size(0), data.size(0))
    
    return losses.avg, accs.avg * 100


# ============================================
# 辅助函数
# ============================================

def get_gpu_memory():
    """获取当前GPU显存使用量(MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def print_training_info(epoch, epochs, train_loss, train_acc, test_loss, test_acc,
                        epoch_time, throughput, gpu_memory):
    """打印训练信息"""
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
          f"Time: {epoch_time:.2f}s | "
          f"Throughput: {throughput:.1f} samples/s | "
          f"GPU Mem: {gpu_memory:.0f}MB")


def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(state, filepath)