"""
实验: Parameter Server 模式 vs Collective 模式
使用 torch.distributed.rpc 实现真正的 Parameter Server 架构
支持 NVTX 性能分析

架构:
- Rank 0: Parameter Server (持有模型参数)
- Rank 1, 2, ...: Workers (计算梯度)
"""

import os
import time
import argparse
import threading
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch import Tensor
import torch.cuda.nvtx as nvtx

import config
from data_loader import get_dataloaders
from model_factory import get_model, count_parameters
from utils import AverageMeter, MetricsLogger, NVTXRange


# ============================================
# Parameter Server 实现
# ============================================

class ParameterServer:
    """
    参数服务器
    - 持有模型参数
    - 接收并聚合来自 workers 的梯度
    - 更新参数并返回给 workers
    """
    def __init__(self, model_name, lr=0.01, num_workers=1, device=None):
        self.device = device if device else torch.device('cpu')
        self.model = get_model(model_name).to(self.device)
        self.num_workers = num_workers
        self.lr = lr
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 梯度聚合缓存
        self.grad_cache = defaultdict(list)
        self.lock = threading.Lock()
        self.update_count = 0
        
        print(f"[PS] Parameter Server initialized with {model_name} on {self.device}")
        total_params, _ = count_parameters(self.model)
        print(f"[PS] Model parameters: {total_params:,}")
    
    def get_parameters(self):
        """返回当前模型参数（供 worker 拉取，转到CPU传输）"""
        return {name: param.data.cpu().clone() for name, param in self.model.named_parameters()}
    
    def push_gradients(self, grads, worker_id):
        """
        接收 worker 的梯度并聚合
        当收集到所有 worker 的梯度后，执行参数更新
        """
        with self.lock:
            # 缓存梯度（先保存在CPU上）
            for name, grad in grads.items():
                self.grad_cache[name].append(grad)
            
            # 检查是否收集完所有 worker 的梯度
            first_key = list(self.grad_cache.keys())[0]
            if len(self.grad_cache[first_key]) == self.num_workers:
                # 聚合梯度（平均）并移到PS的设备上
                self.optimizer.zero_grad()
                for name, param in self.model.named_parameters():
                    if name in self.grad_cache:
                        avg_grad = torch.stack(self.grad_cache[name]).mean(dim=0).to(self.device)
                        param.grad = avg_grad
                
                # 更新参数
                self.optimizer.step()
                self.update_count += 1
                
                # 清空缓存
                self.grad_cache.clear()
                
                return True
        return False
    
    def get_update_count(self):
        return self.update_count


# 全局 PS 实例（用于 RPC 调用）
PARAMETER_SERVER = None


def init_ps(model_name, lr, num_workers, device=None):
    """初始化全局 Parameter Server"""
    global PARAMETER_SERVER
    PARAMETER_SERVER = ParameterServer(model_name, lr, num_workers, device)


def ps_get_parameters():
    """RPC 接口: 获取参数"""
    return PARAMETER_SERVER.get_parameters()


def ps_push_gradients(grads, worker_id):
    """RPC 接口: 推送梯度"""
    return PARAMETER_SERVER.push_gradients(grads, worker_id)


def ps_get_update_count():
    """RPC 接口: 获取更新次数"""
    return PARAMETER_SERVER.get_update_count()


# ============================================
# Worker 实现
# ============================================

class Worker:
    """
    Worker 节点
    - 从 PS 拉取参数
    - 本地计算梯度
    - 将梯度推送到 PS
    """
    def __init__(self, ps_name, worker_id, model_name, device, profile=False):
        self.ps_name = ps_name
        self.worker_id = worker_id
        self.device = device
        self.profile = profile
        
        # 本地模型（用于前向和反向计算）
        self.model = get_model(model_name).to(device)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"[Worker {worker_id}] Initialized on {device}")
    
    def pull_parameters(self):
        """从 PS 拉取最新参数"""
        with NVTXRange("PullParams_RPC", enabled=self.profile):
            params = rpc_sync(self.ps_name, ps_get_parameters)
        
        with NVTXRange("PullParams_Copy", enabled=self.profile):
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in params:
                        param.copy_(params[name].to(self.device))
    
    def compute_and_push_gradients(self, data, target):
        """计算梯度并推送到 PS"""
        with NVTXRange("DataToDevice", enabled=self.profile):
            data, target = data.to(self.device), target.to(self.device)
        
        with NVTXRange("ZeroGrad", enabled=self.profile):
            self.model.zero_grad()
        
        # 前向传播
        with NVTXRange("Forward_PS", enabled=self.profile):
            output = self.model(data)
        
        with NVTXRange("Loss", enabled=self.profile):
            loss = self.criterion(output, target)
        
        # 反向传播
        with NVTXRange("Backward_PS", enabled=self.profile):
            loss.backward()
        
        # 收集梯度
        with NVTXRange("CollectGrads", enabled=self.profile):
            grads = {name: param.grad.cpu() for name, param in self.model.named_parameters() 
                     if param.grad is not None}
        
        # 推送到 PS
        with NVTXRange("PushGrads_RPC", enabled=self.profile):
            rpc_sync(self.ps_name, ps_push_gradients, args=(grads, self.worker_id))
        
        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        return loss.item(), correct, data.size(0)


# ============================================
# 训练流程
# ============================================

def run_ps(args):
    """Parameter Server 进程"""
    device = torch.device('cuda:0') if args.ps_on_gpu else torch.device('cpu')
    print(f"[PS] Starting Parameter Server on {device}...")
    
    # 初始化 PS
    init_ps(args.model, args.lr, args.num_workers, device)
    
    # PS 等待所有 worker 完成
    total_updates = args.epochs * (50000 // (args.batch_size * args.num_workers))
    
    while PARAMETER_SERVER.get_update_count() < total_updates:
        time.sleep(1)
    
    print(f"[PS] Training complete. Total updates: {PARAMETER_SERVER.get_update_count()}")


def run_worker(rank, args):
    """Worker 进程"""
    worker_id = rank - 1  # rank 0 是 PS，worker 从 1 开始
    
    # 每个worker使用对应的GPU（如果PS在GPU0，worker从GPU1开始）
    if args.ps_on_gpu:
        gpu_id = worker_id + 1  # GPU0给PS，worker用GPU1,2,...
    else:
        gpu_id = worker_id  # PS在CPU，worker用GPU0,1,...
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    print(f"[Worker {worker_id}] Starting on {device}...")
    if args.profile:
        print(f"[Worker {worker_id}] NVTX profiling enabled")
    
    # 创建 Worker
    worker = Worker("ps", worker_id, args.model, device, profile=args.profile)
    
    # 加载数据
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        distributed=False,
        use_cache=args.use_cache
    )
    
    # 日志（只在 worker 0 记录）
    if worker_id == 0:
        exp_name = f"ps_mode_{args.model}_bs{args.batch_size}_workers{args.num_workers}"
        logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    best_acc = 0
    
    for epoch in range(args.epochs):
        with NVTXRange(f"Epoch_{epoch}", enabled=args.profile):
            worker.model.train()
            losses = AverageMeter()
            accs = AverageMeter()
            
            start_time = time.time()
            total_samples_epoch = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # 简单的数据分片：每个 worker 处理不同的 batch
                if batch_idx % args.num_workers != worker_id:
                    continue
                
                with NVTXRange("Batch", enabled=args.profile):
                    # 拉取最新参数
                    worker.pull_parameters()
                    
                    # 计算梯度并推送
                    loss, correct, batch_size = worker.compute_and_push_gradients(data, target)
                    
                    # 等待参数更新完成（同步点）
                    with NVTXRange("WaitSync", enabled=args.profile):
                        torch.cuda.synchronize(device)
                
                losses.update(loss, batch_size)
                accs.update(correct / batch_size, batch_size)
                total_samples_epoch += batch_size
            
            epoch_time = time.time() - start_time
            throughput = total_samples_epoch / epoch_time
            
            # 评估（只在 worker 0 上进行）
            if worker_id == 0:
                with NVTXRange("Evaluation", enabled=args.profile):
                    worker.pull_parameters()  # 拉取最新参数
                    worker.model.eval()
                    test_losses = AverageMeter()
                    test_accs = AverageMeter()
                    
                    with torch.no_grad():
                        for data, target in test_loader:
                            with NVTXRange("EvalBatch", enabled=args.profile):
                                data, target = data.to(device), target.to(device)
                                output = worker.model(data)
                                loss = worker.criterion(output, target)
                                pred = output.argmax(dim=1, keepdim=True)
                                correct = pred.eq(target.view_as(pred)).sum().item()
                                test_losses.update(loss.item(), data.size(0))
                                test_accs.update(correct / data.size(0), data.size(0))
                
                train_acc = accs.avg * 100
                test_acc = test_accs.avg * 100
                gpu_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0
                
                # 汇总所有 worker 的吞吐量
                total_throughput = throughput * args.num_workers
                
                print(f"[Worker 0] Epoch [{epoch+1}/{args.epochs}] "
                      f"Train Loss: {losses.avg:.4f} Acc: {train_acc:.2f}% | "
                      f"Test Loss: {test_losses.avg:.4f} Acc: {test_acc:.2f}% | "
                      f"Time: {epoch_time:.2f}s | Throughput: {total_throughput:.1f} samples/s")
                
                logger.log(epoch, losses.avg, train_acc, test_losses.avg, test_acc,
                           epoch_time, total_throughput, gpu_memory)
                
                if test_acc > best_acc:
                    best_acc = test_acc
    
    if worker_id == 0:
        extra_info = {
            'model': args.model,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'best_acc': best_acc,
            'training_type': 'parameter_server',
            'communication_mode': 'ps'
        }
        logger.save(extra_info)
        print(f"\n[Worker 0] Best Test Accuracy: {best_acc:.2f}%")


def run(rank, world_size, args):
    """主运行函数"""
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = str(int(config.MASTER_PORT) + 100)  # 使用不同端口避免冲突
    
    # RPC 初始化选项
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=300
    )
    
    if rank == 0:
        # Parameter Server
        rpc.init_rpc(
            name="ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        run_ps(args)
    else:
        # Worker
        rpc.init_rpc(
            name=f"worker_{rank-1}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        run_worker(rank, args)
    
    rpc.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Parameter Server Training')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers (excluding PS)')
    parser.add_argument('--ps_on_gpu', action='store_true', help='Run PS on GPU0 (workers on GPU1,2,...)')
    parser.add_argument('--profile', action='store_true', help='Enable NVTX profiling')
    
    args = parser.parse_args()
    
    # world_size = 1 (PS) + num_workers
    world_size = 1 + args.num_workers
    
    print(f"Starting Parameter Server training with {args.num_workers} workers")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, Epochs: {args.epochs}")
    if args.ps_on_gpu:
        print(f"PS on GPU0, Workers on GPU1-{args.num_workers}")
    else:
        print(f"PS on CPU, Workers on GPU0-{args.num_workers-1}")
    
    import torch.multiprocessing as mp
    mp.spawn(
        run,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()