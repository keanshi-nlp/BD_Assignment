"""数据加载模块 - CIFAR-100 fine_label"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import config


def get_transforms():
    """数据增强和预处理"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    return train_transform, test_transform


def get_dataloaders(batch_size=config.BATCH_SIZE, distributed=False, rank=0, world_size=1, use_cache=False):
    """
    获取数据加载器
    
    Args:
        batch_size: 批次大小
        distributed: 是否分布式训练
        rank: 当前进程rank
        world_size: 总进程数
        use_cache: 是否使用缓存（将数据加载到内存）
    """
    train_transform, test_transform = get_transforms()
    
    # 加载CIFAR-100数据集 (默认使用fine_label，100个类别)
    train_dataset = datasets.CIFAR100(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=config.DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 缓存数据到内存
    if use_cache:
        train_dataset = CachedDataset(train_dataset)
        test_dataset = CachedDataset(test_dataset)
    
    # 分布式采样器
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler


class CachedDataset(torch.utils.data.Dataset):
    """将数据集缓存到内存中"""
    def __init__(self, dataset):
        self.data = []
        self.targets = []
        print("Caching dataset to memory...")
        for i in range(len(dataset)):
            img, label = dataset[i]
            self.data.append(img)
            self.targets.append(label)
        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        print(f"Cached {len(self.data)} samples")
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)