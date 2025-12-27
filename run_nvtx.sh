#!/bin/bash
# ============================================
# Nsight Systems 性能分析脚本
# 使用 NVTX 标记分析各训练模式的性能
# ============================================

set -e

echo "============================================"
echo "Nsight Systems 性能分析"
echo "============================================"

export CUDA_VISIBLE_DEVICES=0,1,2

# 配置
MODEL="resnet18"
BATCH_SIZE=256
EPOCHS=10  # 分析只需要几个epoch

# 创建输出目录
mkdir -p nsight_reports logs checkpoints

# 检查nsys是否可用
if ! command -v nsys &> /dev/null; then
    echo "警告: nsys 未找到，将只运行带 --profile 参数的训练（不生成报告）"
    USE_NSYS=false
else
    USE_NSYS=true
    echo "nsys 版本: $(nsys --version)"
fi

# ============================================
# 1. Single GPU 分析
# ============================================
# echo ""
# echo "========== [1/9] Single GPU Profiling =========="
# if $USE_NSYS; then
#     nsys profile \
#         --trace=cuda,cudnn,cublas,nvtx,osrt \
#         --sample=none \
#         --output=nsight_reports/prof_single_gpu \
#         --force-overwrite=true \
#         --stats=true \
#         python train_single_gpu.py \
#             --model $MODEL \
#             --batch_size $BATCH_SIZE \
#             --epochs $EPOCHS \
#             --profile
# else
#     python train_single_gpu.py \
#         --model $MODEL \
#         --batch_size $BATCH_SIZE \
#         --epochs $EPOCHS \
#         --profile
# fi

# # ============================================
# # 2. DataParallel 分析
# # ============================================
# echo ""
# echo "========== [2/9] DataParallel Profiling =========="
# if $USE_NSYS; then
#     nsys profile \
#         --trace=cuda,cudnn,cublas,nvtx,osrt \
#         --sample=none \
#         --output=nsight_reports/prof_dp \
#         --force-overwrite=true \
#         --stats=true \
#         python train_dp.py \
#             --model $MODEL \
#             --batch_size $BATCH_SIZE \
#             --epochs $EPOCHS \
#             --profile
# else
#     python train_dp.py \
#         --model $MODEL \
#         --batch_size $BATCH_SIZE \
#         --epochs $EPOCHS \
#         --profile
# fi

# # ============================================
# # 3. DDP (NCCL) 分析
# # ============================================
# echo ""
# echo "========== [3/9] DDP (NCCL) Profiling =========="
# if $USE_NSYS; then
#     nsys profile \
#         --trace=cuda,cudnn,cublas,nvtx,osrt \
#         --sample=none \
#         --output=nsight_reports/prof_ddp_nccl \
#         --force-overwrite=true \
#         --stats=true \
#         python train_ddp.py \
#             --model $MODEL \
#             --batch_size $((BATCH_SIZE / 2)) \
#             --epochs $EPOCHS \
#             --backend nccl \
#             --profile
# else
#     python train_ddp.py \
#         --model $MODEL \
#         --batch_size $((BATCH_SIZE / 2)) \
#         --epochs $EPOCHS \
#         --backend nccl \
#         --profile
# fi

# # ============================================
# # 4. DDP (Gloo) 分析
# # ============================================
# echo ""
# echo "========== [4/9] DDP (Gloo) Profiling =========="
# if $USE_NSYS; then
#     nsys profile \
#         --trace=cuda,cudnn,cublas,nvtx,osrt \
#         --sample=none \
#         --output=nsight_reports/prof_ddp_gloo \
#         --force-overwrite=true \
#         --stats=true \
#         python train_ddp.py \
#             --model $MODEL \
#             --batch_size $((BATCH_SIZE / 2)) \
#             --epochs $EPOCHS \
#             --backend gloo \
#             --profile
# else
#     python train_ddp.py \
#         --model $MODEL \
#         --batch_size $((BATCH_SIZE / 2)) \
#         --epochs $EPOCHS \
#         --backend gloo \
#         --profile
# fi

# # ============================================
# # 5. FSDP 分析
# # ============================================
# echo ""
# echo "========== [5/9] FSDP Profiling =========="
# if $USE_NSYS; then
#     nsys profile \
#         --trace=cuda,cudnn,cublas,nvtx,osrt \
#         --sample=none \
#         --output=nsight_reports/prof_fsdp \
#         --force-overwrite=true \
#         --stats=true \
#         python train_fsdp.py \
#             --model $MODEL \
#             --batch_size $((BATCH_SIZE / 2)) \
#             --epochs $EPOCHS \
#             --sharding full \
#             --profile
# else
#     python train_fsdp.py \
#         --model $MODEL \
#         --batch_size $((BATCH_SIZE / 2)) \
#         --epochs $EPOCHS \
#         --sharding full \
#         --profile
# fi

# # ============================================
# # 6. Model Parallel 分析
# # ============================================
# echo ""
# echo "========== [6/9] Model Parallel Profiling =========="
# if $USE_NSYS; then
#     nsys profile \
#         --trace=cuda,cudnn,cublas,nvtx,osrt \
#         --sample=none \
#         --output=nsight_reports/prof_model_parallel \
#         --force-overwrite=true \
#         --stats=true \
#         python train_model_parallel.py \
#             --model $MODEL \
#             --batch_size $BATCH_SIZE \
#             --epochs $EPOCHS \
#             --profile
# else
#     python train_model_parallel.py \
#         --model $MODEL \
#         --batch_size $BATCH_SIZE \
#         --epochs $EPOCHS \
#         --profile
# fi

# ============================================
# 7. Collective (NCCL AllReduce) 分析
# ============================================
echo ""
echo "========== [7/9] Collective (NCCL) Profiling =========="
if $USE_NSYS; then
    nsys profile \
        --trace=cuda,cudnn,cublas,nvtx,osrt \
        --sample=none \
        --output=nsight_reports/prof_collective_nccl \
        --force-overwrite=true \
        --stats=true \
        python train_collective.py \
            --model $MODEL \
            --batch_size $((BATCH_SIZE / 2)) \
            --epochs $EPOCHS \
            --backend nccl \
            --profile
else
    python train_collective.py \
        --model $MODEL \
        --batch_size $((BATCH_SIZE / 2)) \
        --epochs $EPOCHS \
        --backend nccl \
        --profile
fi

# ============================================
# 8. Parameter Server (PS on CPU) 分析
# ============================================
echo ""
echo "========== [8/9] Parameter Server (CPU) Profiling =========="
if $USE_NSYS; then
    nsys profile \
        --trace=cuda,cudnn,cublas,nvtx,osrt \
        --sample=none \
        --output=nsight_reports/prof_ps_cpu \
        --force-overwrite=true \
        --stats=true \
        python train_ps.py \
            --model $MODEL \
            --batch_size $((BATCH_SIZE / 2)) \
            --epochs $EPOCHS \
            --num_workers 2 \
            --profile
else
    python train_ps.py \
        --model $MODEL \
        --batch_size $((BATCH_SIZE / 2)) \
        --epochs $EPOCHS \
        --num_workers 2 \
        --profile
fi

# ============================================
# 9. Parameter Server (PS on GPU) 分析
# ============================================
echo ""
echo "========== [9/9] Parameter Server (GPU) Profiling =========="
if $USE_NSYS; then
    nsys profile \
        --trace=cuda,cudnn,cublas,nvtx,osrt \
        --sample=none \
        --output=nsight_reports/prof_ps_gpu \
        --force-overwrite=true \
        --stats=true \
        python train_ps.py \
            --model $MODEL \
            --batch_size $((BATCH_SIZE / 2)) \
            --epochs $EPOCHS \
            --num_workers 2 \
            --ps_on_gpu \
            --profile
else
    python train_ps.py \
        --model $MODEL \
        --batch_size $((BATCH_SIZE / 2)) \
        --epochs $EPOCHS \
        --num_workers 2 \
        --ps_on_gpu \
        --profile
fi

# ============================================
# 生成统计摘要
# ============================================
echo ""
echo "============================================"
echo "分析完成!"
echo "============================================"

if $USE_NSYS; then
    echo ""
    echo "生成的报告文件:"
    ls -lh nsight_reports/*.nsys-rep 2>/dev/null || echo "  (无报告文件)"
    
    echo ""
    echo "查看报告的方法:"
    echo "  1. GUI方式: nsys-ui nsight_reports/prof_xxx.nsys-rep"
    echo "  2. 命令行统计: nsys stats nsight_reports/prof_xxx.nsys-rep"
    echo ""
    echo "生成统计摘要..."
    for report in nsight_reports/*.nsys-rep; do
        if [ -f "$report" ]; then
            name=$(basename "$report" .nsys-rep)
            echo ""
            echo "=== $name ==="
            nsys stats "$report" --report nvtx_sum 2>/dev/null | head -30 || echo "  (无NVTX统计)"
        fi
    done
fi

echo ""
echo "============================================"
echo "NVTX 标记说明:"
echo "============================================"
echo "| 标记名称                | 说明                      |"
echo "|-------------------------|---------------------------|"
echo "| Batch                   | 完整的一个batch训练       |"
echo "| DataToDevice            | 数据从CPU传输到GPU        |"
echo "| ZeroGrad                | 梯度清零                  |"
echo "| Forward                 | 前向传播                  |"
echo "| Forward_AMP             | 混合精度前向传播          |"
echo "| Forward_DP_Scatter      | DP前向(数据分发)          |"
echo "| Forward_FSDP_AllGather  | FSDP前向(参数聚合)        |"
echo "| Forward_Pipeline        | 模型并行前向              |"
echo "| Forward_PS              | PS模式前向                |"
echo "| Loss                    | 损失计算                  |"
echo "| Backward                | 反向传播                  |"
echo "| Backward_DP_Gather      | DP反向(梯度汇总)          |"
echo "| Backward_AllReduce      | DDP反向(含AllReduce)      |"
echo "| Backward_FSDP_*         | FSDP反向(ReduceScatter)   |"
echo "| Backward_Pipeline       | 模型并行反向              |"
echo "| Backward_PS             | PS模式反向                |"
echo "| OptimizerStep           | 优化器更新参数            |"
echo "| PullParams_RPC          | PS: RPC拉取参数           |"
echo "| PushGrads_RPC           | PS: RPC推送梯度           |"
echo "| CollectGrads            | PS: 收集梯度到CPU         |"
echo "| WaitSync                | 等待同步                  |"
echo "| Evaluation              | 模型评估                  |"
echo "============================================"

echo ""
echo "实验配置汇总:"
echo "============================================"
echo "| 实验                    | Batch Size | GPUs        |"
echo "|-------------------------|------------|-------------|"
echo "| Single GPU              | $BATCH_SIZE        | 1           |"
echo "| DataParallel            | $BATCH_SIZE        | 2 (split)   |"
echo "| DDP (NCCL/Gloo)         | $((BATCH_SIZE/2))x2=$BATCH_SIZE    | 2           |"
echo "| FSDP                    | $((BATCH_SIZE/2))x2=$BATCH_SIZE    | 2           |"
echo "| Model Parallel          | $BATCH_SIZE        | 2 (pipeline)|"
echo "| Collective              | $((BATCH_SIZE/2))x2=$BATCH_SIZE    | 2           |"
echo "| PS (CPU)                | $((BATCH_SIZE/2))x2=$BATCH_SIZE    | 2 workers   |"
echo "| PS (GPU)                | $((BATCH_SIZE/2))x2=$BATCH_SIZE    | 3 (1PS+2W)  |"
echo "============================================"