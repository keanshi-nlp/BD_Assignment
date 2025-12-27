#!/bin/bash
# ============================================
# Parameter Server vs Collective 模式对比实验
# ============================================
export CUDA_VISIBLE_DEVICES=0,1,2
set -e

echo "============================================"
echo "通信模式对比实验: PS vs Collective"
echo "============================================"

mkdir -p logs checkpoints plots

# ============================================
# 实验配置
# ============================================
MODEL="resnet18"
BATCH_SIZE=128
EPOCHS=50
LR=0.01

# ============================================
# 实验1: Collective 模式 (AllReduce - NCCL)
# ============================================
# echo ""
# echo "========== [1] Collective Mode (NCCL AllReduce) =========="
# python train_collective.py \
#     --model $MODEL \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --backend nccl

# # ============================================
# # 实验2: Collective 模式 (AllReduce - Gloo)
# # ============================================
# echo ""
# echo "========== [2] Collective Mode (Gloo AllReduce) =========="
# python train_collective.py \
#     --model $MODEL \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --backend gloo

# # ============================================
# # 实验3: Parameter Server 模式 (2 workers)
# # ============================================
# echo ""
# echo "========== [3] Parameter Server Mode (2 workers) =========="
# python train_ps.py \
#     --model $MODEL \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --num_workers 2

python train_ps_gpu.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --num_workers 2

# ============================================
# 结果分析
# ============================================
echo ""
echo "========== 分析结果 =========="

echo ""
echo "实验完成！结果对比："
echo ""
echo "| 模式 | 说明 |"
echo "|------|------|"
echo "| Collective (NCCL) | Ring-AllReduce, GPU优化 |"
echo "| Collective (Gloo) | Ring-AllReduce, 通用 |"
echo "| Parameter Server | 中心化参数服务器 |"
echo ""
echo "查看详细结果: ./logs/"
echo "============================================"