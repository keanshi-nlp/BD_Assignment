"""实验配置"""

import os

# 数据配置
DATA_ROOT = './data'
NUM_CLASSES = 100
NUM_WORKERS = 4

# 训练配置
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg16', 'densenet121'
MODEL_NAME = 'resnet18'

# 分布式配置
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

# 日志和检查点
LOG_DIR = './logs'
CHECKPOINT_DIR = './checkpoints'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)