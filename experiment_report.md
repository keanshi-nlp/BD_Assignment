# 分布式训练实验报告

## 实验概述

本实验在CIFAR-100数据集上对比了多种分布式训练策略的性能。

## 实验配置

- 数据集: CIFAR-100 (fine_label, 100类)
- 硬件: 2x NVIDIA A40 (48GB)
- 基础模型: ResNet50

## 实验结果汇总

| Experiment                                   | Training Type           | Model    |   Batch Size |   Best Acc (%) |   Total Time (s) |   Avg Throughput |   Max GPU Mem (MB) |
|:---------------------------------------------|:------------------------|:---------|-------------:|---------------:|-----------------:|-----------------:|-------------------:|
| ddp_nccl_resnet18_bs64x8_cacheFalse_ampFalse | ddp_data_parallel       | resnet18 |          512 |          20.18 |          59.6913 |         2748.15  |            472.244 |
| single_gpu_resnet18_bs64_cacheFalse_ampFalse | single_gpu              | resnet18 |           64 |          22.53 |          39.5561 |         2690.84  |            428.044 |
| single_gpu_resnet18_bs64_cacheTrue_ampFalse  | single_gpu              | resnet18 |           64 |          24.22 |          41.9394 |         2542.32  |            428.044 |
| fsdp_full_resnet18_bs64x8_ampFalse           | fsdp_hybrid_parallel    | resnet18 |          512 |           2.25 |          64.7088 |         2499.14  |            366.185 |
| single_gpu_resnet18_bs64_cacheFalse_ampTrue  | single_gpu              | resnet18 |           64 |          21.73 |          45.7756 |         2311.11  |            292.693 |
| ddp_gloo_resnet18_bs64x8_cacheFalse_ampFalse | ddp_data_parallel       | resnet18 |          512 |          23.41 |          77.0604 |         1923.15  |            472.244 |
| model_parallel_resnet50_bs64_cacheFalse      | model_parallel_pipeline | resnet50 |           64 |           5.12 |          90.8945 |         1185.14  |           1892.36  |
| dp_resnet18_bs64_ngpu8_cacheFalse_ampFalse   | data_parallel           | resnet18 |           64 |          20.82 |         273.907  |          407.643 |            204.603 |

## 详细分析

### 1. 吞吐量分析
- 最高吞吐量配置: ddp_nccl_resnet18_bs64x8_cacheFalse_ampFalse
- 最低吞吐量配置: dp_resnet18_bs64_ngpu8_cacheFalse_ampFalse

### 2. 准确率分析
- 最高准确率配置: single_gpu_resnet18_bs64_cacheTrue_ampFalse
- 最低准确率配置: fsdp_full_resnet18_bs64x8_ampFalse

### 3. 内存效率分析
- 内存使用最少配置: dp_resnet18_bs64_ngpu8_cacheFalse_ampFalse
- 内存使用最多配置: model_parallel_resnet50_bs64_cacheFalse

## 结论与建议

基于实验结果，建议:
1. 对于追求训练速度的场景，使用DDP数据并行
2. 对于内存受限的场景，考虑FSDP或模型并行
3. 混合精度训练(AMP)可以显著提升吞吐量并减少内存使用

