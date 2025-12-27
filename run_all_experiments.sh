#!/bin/bash
# ============================================
# 分布式训练对比实验 - 完整实验脚本
# 硬件: 2x A40 48GB
# 数据集: CIFAR-100 (fine_label)
# ============================================
export CUDA_VISIBLE_DEVICES=0,1

set -e

echo "============================================"
echo "分布式训练对比实验"
echo "============================================"

# 创建必要目录
mkdir -p logs checkpoints plots

# ============================================
# 实验组1: 单机单卡 vs 单机多卡
# ============================================
echo ""
echo "========== 实验组1: 单卡 vs 多卡 =========="

# 1.1 单卡训练 (基线)
echo "[1.1] Single GPU Training (baseline)..."
python train_single_gpu.py --model resnet34 --batch_size 256 --epochs 50 --gpu 0

# 1.2 DataParallel (DP)
echo "[1.2] DataParallel Training..."
python train_dp.py --model resnet34 --batch_size 256 --epochs 50

# 1.3 DistributedDataParallel (DDP) - NCCL
echo "[1.3] DDP Training (NCCL backend)..."
python train_ddp.py --model resnet34 --batch_size 128 --epochs 50 --backend nccl --scale_lr

# ============================================
# 实验组2: 不同后端对比 (NCCL vs Gloo)
# ============================================
echo ""
echo "========== 实验组2: 后端对比 =========="

# 2.1 DDP with NCCL (已在1.3完成)

# 2.2 DDP with Gloo
echo "[2.2] DDP Training (Gloo backend)..."
python train_ddp.py --model resnet34 --batch_size 128 --epochs 50 --backend gloo --scale_lr

# ============================================
# 实验组3: 不同并行策略对比
# ============================================
echo ""
echo "========== 实验组3: 并行策略对比 =========="

# 3.1 数据并行 (DDP) - 已完成

# 3.2 模型并行 (Pipeline)
echo "[3.2] Model Parallel (Pipeline)..."
python train_model_parallel.py --model resnet34 --batch_size 256 --epochs 50

3.3 FSDP (混合并行 - Full Shard)
echo "[3.3] FSDP (Full Shard)..."
python train_fsdp.py --model resnet34 --batch_size 128 --epochs 50 --sharding full

# 3.4 FSDP (Gradient Shard Only)
echo "[3.4] FSDP (Gradient Shard)..."
python train_fsdp.py --model resnet34 --batch_size 128 --epochs 50 --sharding grad

# ============================================
# 实验组4: 缓存对比
# ============================================
echo ""
echo "========== 实验组4: 缓存对比 =========="

# 4.1 无缓存 (已在之前实验完成)

# 4.2 有缓存 - 单卡
echo "[4.2] Single GPU with Cache..."
python train_single_gpu.py --model resnet34 --batch_size 256 --epochs 50 --gpu 0 --use_cache

# 4.3 有缓存 - DDP
echo "[4.3] DDP with Cache..."
python train_ddp.py --model resnet34 --batch_size 128 --epochs 50 --backend nccl --use_cache --scale_lr

# ============================================
# 实验组5: 混合精度(AMP)对比
# ============================================
echo ""
echo "========== 实验组5: AMP对比 =========="

# 5.1 单卡 + AMP
echo "[5.1] Single GPU with AMP..."
python train_single_gpu.py --model resnet34 --batch_size 256 --epochs 50 --gpu 0 --amp

# 5.2 DDP + AMP
echo "[5.2] DDP with AMP..."
python train_ddp.py --model resnet34 --batch_size 128 --epochs 50 --backend nccl --amp --scale_lr

# 5.3 单卡 + Cache + AMP (最优配置)
echo "[5.3] Single GPU with Cache + AMP..."
python train_single_gpu.py --model resnet34 --batch_size 256 --epochs 50 --gpu 0 --use_cache --amp

# 5.4 DDP + Cache + AMP (最优配置)
echo "[5.4] DDP with Cache + AMP..."
python train_ddp.py --model resnet34 --batch_size 128 --epochs 50 --backend nccl --use_cache --amp --scale_lr

# ============================================
# 实验组6: DDP 下不同Batch Size对比
# ============================================
echo ""
echo "========== 实验组6: Batch Size对比 =========="

# 6.1 Small batch
echo "[6.1] DDP with small batch (64)..."
python train_ddp.py --model resnet34 --batch_size 64 --epochs 50 --backend nccl --scale_lr

# 6.2 Large batch
echo "[6.2] DDP with large batch (256)..."
python train_ddp.py --model resnet34 --batch_size 256 --epochs 50 --backend nccl --scale_lr

# ============================================
# 实验组7: 不同模型对比 (可选)
# ============================================
# echo ""
# echo "========== 实验组7: 不同模型对比 =========="

# # 7.1 ResNet34 (轻量)
# echo "[7.1] ResNet34..."
# python train_ddp.py --model resnet34 --batch_size 256 --epochs 50 --backend nccl --scale_lr

# # 7.2 ResNet50 (重量)
# echo "[7.2] ResNet50..."
# python train_ddp.py --model resnet50 --batch_size 256 --epochs 50 --backend nccl --scale_lr

# ============================================
# 实验组8: single gpu 下不同Batch Size对比
# ============================================
echo "========== 实验组8: bsz 对比 =========="
echo "[8.1] single gpu small batch (128)"
python train_single_gpu.py --model resnet34 --batch_size 128 --epochs 50 --gpu 0

echo "[8.2] single gpu large batch (512)"
python train_single_gpu.py --model resnet34 --batch_size 512 --epochs 50 --gpu 0

# ============================================
# 实验组9: 模式对比
# ============================================
echo "========== 实验组8: bsz 对比 =========="
echo "[8.1] single gpu small batch (128)"
python train_single_gpu.py --model resnet34 --batch_size 128 --epochs 50 --gpu 0

echo "[8.2] single gpu large batch (512)"
python train_single_gpu.py --model resnet34 --batch_size 512 --epochs 50 --gpu 0

# ============================================
# 结果分析
# ============================================
echo ""
echo "========== 分析实验结果 =========="
python analyze_results.py

echo ""
echo "============================================"
echo "所有实验完成!"
echo "结果保存在: ./logs/"
echo "图表保存在: ./plots/"
echo "报告保存在: ./experiment_report.md"
echo "============================================"