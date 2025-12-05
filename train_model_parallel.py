"""
实验4: 模型并行 - 流水线并行
将模型分割到多个GPU上
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse

import config
from data_loader import get_dataloaders
from utils import AverageMeter, get_gpu_memory, MetricsLogger


class PipelineResNet(nn.Module):
    """
    流水线并行的ResNet
    将模型分成两部分，分别放在两个GPU上
    """
    def __init__(self, num_classes=100, device0='cuda:0', device1='cuda:1'):
        super(PipelineResNet, self).__init__()
        self.device0 = device0
        self.device1 = device1
        
        # 第一部分：在GPU0上
        # 使用ResNet的前半部分
        from torchvision.models import resnet50
        base_model = resnet50(pretrained=False)
        
        # 修改第一层适配CIFAR
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        
        self.part1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
        ).to(device0)
        
        # 第二部分：在GPU1上
        self.part2 = nn.Sequential(
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        ).to(device1)
        
        self.fc = nn.Linear(2048, num_classes).to(device1)
        
    def forward(self, x):
        x = x.to(self.device0)
        x = self.part1(x)
        x = x.to(self.device1)
        x = self.part2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PipelineVGG(nn.Module):
    """流水线并行的VGG16"""
    def __init__(self, num_classes=100, device0='cuda:0', device1='cuda:1'):
        super(PipelineVGG, self).__init__()
        self.device0 = device0
        self.device1 = device1
        
        from torchvision.models import vgg16_bn
        base_model = vgg16_bn(pretrained=False)
        
        # 分割features
        features = list(base_model.features.children())
        mid = len(features) // 2
        
        self.part1 = nn.Sequential(*features[:mid]).to(device0)
        self.part2 = nn.Sequential(*features[mid:]).to(device1)
        
        self.avgpool = base_model.avgpool.to(device1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ).to(device1)
        
    def forward(self, x):
        x = x.to(self.device0)
        x = self.part1(x)
        x = x.to(self.device1)
        x = self.part2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_epoch_pipeline(model, train_loader, criterion, optimizer, device1):
    """流水线并行训练一个epoch"""
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    start_time = time.time()
    total_samples = 0
    
    for data, target in train_loader:
        target = target.to(device1)
        
        optimizer.zero_grad()
        output = model(data)  # 模型内部处理device转移
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        losses.update(loss.item(), data.size(0))
        accs.update(correct / data.size(0), data.size(0))
        total_samples += data.size(0)
    
    epoch_time = time.time() - start_time
    throughput = total_samples / epoch_time
    
    return losses.avg, accs.avg * 100, epoch_time, throughput


def evaluate_pipeline(model, test_loader, criterion, device1):
    """评估流水线模型"""
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device1)
            output = model(data)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            
            losses.update(loss.item(), data.size(0))
            accs.update(correct / data.size(0), data.size(0))
    
    return losses.avg, accs.avg * 100


def train_model_parallel(args):
    """模型并行训练"""
    device0 = 'cuda:0'
    device1 = 'cuda:1'
    
    print(f"Model Parallel Training: {args.model}")
    print(f"Part1 on {device0}, Part2 on {device1}")
    
    # 加载数据
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        distributed=False,
        use_cache=args.use_cache
    )
    
    # 创建流水线模型
    if args.model == 'resnet50':
        model = PipelineResNet(num_classes=100, device0=device0, device1=device1)
    elif args.model == 'vgg16':
        model = PipelineVGG(num_classes=100, device0=device0, device1=device1)
    else:
        raise ValueError(f"Model parallel not implemented for {args.model}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 日志
    exp_name = f"model_parallel_{args.model}_bs{args.batch_size}_cache{args.use_cache}"
    logger = MetricsLogger(config.LOG_DIR, exp_name)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.epochs):
        torch.cuda.reset_peak_memory_stats()
        
        train_loss, train_acc, epoch_time, throughput = train_epoch_pipeline(
            model, train_loader, criterion, optimizer, device1
        )
        test_loss, test_acc = evaluate_pipeline(model, test_loader, criterion, device1)
        
        scheduler.step()
        
        # 获取两个GPU的内存使用
        gpu0_mem = torch.cuda.max_memory_allocated(0) / 1024 / 1024
        gpu1_mem = torch.cuda.max_memory_allocated(1) / 1024 / 1024
        total_mem = gpu0_mem + gpu1_mem
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s | Throughput: {throughput:.1f} samples/s | "
              f"GPU Mem: {gpu0_mem:.0f}MB + {gpu1_mem:.0f}MB = {total_mem:.0f}MB")
        
        logger.log(epoch, train_loss, train_acc, test_loss, test_acc, 
                   epoch_time, throughput, total_mem)
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    extra_info = {
        'model': args.model,
        'batch_size': args.batch_size,
        'use_cache': args.use_cache,
        'best_acc': best_acc,
        'training_type': 'model_parallel_pipeline'
    }
    logger.save(extra_info)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Parallel Training')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--use_cache', action='store_true')
    
    args = parser.parse_args()
    train_model_parallel(args)