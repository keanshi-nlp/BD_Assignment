"""
NVTX 性能分析工具模块
用于 Nsight Systems 分析
"""

import torch
import torch.cuda.nvtx as nvtx
from contextlib import contextmanager


class NVTXRange:
    """NVTX 范围标记器 (类形式)"""
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
    
    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            nvtx.range_push(self.name)
        return self
    
    def __exit__(self, *args):
        if self.enabled and torch.cuda.is_available():
            nvtx.range_pop()


@contextmanager
def nvtx_range(name, enabled=True):
    """NVTX 范围标记器 (函数形式)"""
    if enabled and torch.cuda.is_available():
        nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled and torch.cuda.is_available():
            nvtx.range_pop()


def nvtx_mark(message, enabled=True):
    """NVTX 单点标记"""
    if enabled and torch.cuda.is_available():
        nvtx.mark(message)


class ProfiledDataLoader:
    """带 NVTX 标记的数据加载器包装"""
    def __init__(self, dataloader, enabled=True):
        self.dataloader = dataloader
        self.enabled = enabled
    
    def __iter__(self):
        for batch_idx, batch in enumerate(self.dataloader):
            if self.enabled and torch.cuda.is_available():
                nvtx.range_push(f"DataBatch_{batch_idx}")
            yield batch
            if self.enabled and torch.cuda.is_available():
                nvtx.range_pop()
    
    def __len__(self):
        return len(self.dataloader)


class ProfiledTrainer:
    """带 NVTX 标记的训练器"""
    def __init__(self, model, criterion, optimizer, device, 
                 scaler=None, use_amp=False, enabled=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.use_amp = use_amp
        self.enabled = enabled
    
    def train_step(self, data, target):
        """单步训练（带 NVTX 标记）"""
        with nvtx_range("DataToDevice", self.enabled):
            data, target = data.to(self.device), target.to(self.device)
        
        with nvtx_range("ZeroGrad", self.enabled):
            self.optimizer.zero_grad()
        
        if self.use_amp and self.scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                with nvtx_range("Forward_AMP", self.enabled):
                    output = self.model(data)
                with nvtx_range("Loss_AMP", self.enabled):
                    loss = self.criterion(output, target)
            
            with nvtx_range("Backward_AMP", self.enabled):
                self.scaler.scale(loss).backward()
            
            with nvtx_range("OptimizerStep_AMP", self.enabled):
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            with nvtx_range("Forward", self.enabled):
                output = self.model(data)
            
            with nvtx_range("Loss", self.enabled):
                loss = self.criterion(output, target)
            
            with nvtx_range("Backward", self.enabled):
                loss.backward()
            
            with nvtx_range("OptimizerStep", self.enabled):
                self.optimizer.step()
        
        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        return loss.item(), correct, data.size(0)