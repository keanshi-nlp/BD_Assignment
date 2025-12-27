#!/bin/bash
# ============================================
# 快速测试脚本 - 验证所有代码可运行
# 每个实验只跑2个epoch
# ============================================

set -e

echo "============================================"
echo "快速测试 (每个实验2个epoch)"
echo "============================================"

mkdir -p logs checkpoints plots

EPOCHS=2
BS=64

# echo "[1/8] Testing Single GPU..."
# python train_single_gpu.py --model resnet18 --batch_size $BS --epochs $EPOCHS --gpu 0

# echo "[2/8] Testing DataParallel..."
# python train_dp.py --model resnet18 --batch_size $BS --epochs $EPOCHS

# echo "[3/8] Testing DDP (NCCL)..."
# python train_ddp.py --model resnet18 --batch_size $BS --epochs $EPOCHS --backend nccl

# echo "[4/8] Testing DDP (Gloo)..."
# python train_ddp.py --model resnet18 --batch_size $BS --epochs $EPOCHS --backend gloo

echo "[5/8] Testing Model Parallel..."
python train_model_parallel.py --model resnet18 --batch_size $BS --epochs $EPOCHS

echo "[6/8] Testing FSDP..."
python train_fsdp.py --model resnet18 --batch_size $BS --epochs $EPOCHS --sharding full

echo "[7/8] Testing with Cache..."
python train_single_gpu.py --model resnet18 --batch_size $BS --epochs $EPOCHS --gpu 0 --use_cache

echo "[8/8] Testing with AMP..."
python train_single_gpu.py --model resnet18 --batch_size $BS --epochs $EPOCHS --gpu 0 --amp

echo ""
echo "============================================"
echo "所有测试通过!"
echo "============================================"

# 分析结果
python analyze_results.py