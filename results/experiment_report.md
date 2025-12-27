# 分布式训练实验报告

## 实验结果汇总

| Experiment                                    | Model    |   Batch Size |   Best Acc (%) |   Avg Throughput |   Max GPU Mem (MB) |   Total Time (s) |
|:----------------------------------------------|:---------|-------------:|---------------:|-----------------:|-------------------:|-----------------:|
| single_gpu_resnet18_bs256_cacheTrue_ampTrue   | resnet18 |          256 |          56.23 |           7868.4 |                729 |            365   |
| single_gpu_resnet18_bs256_cacheFalse_ampTrue  | resnet18 |          256 |          74.22 |           7018.8 |                729 |            405.1 |
| ddp_nccl_resnet18_bs128x2_cacheTrue_ampTrue   | resnet18 |          256 |          66.77 |           5003.9 |                479 |            700.6 |
| single_gpu_resnet18_bs256_cacheTrue_ampFalse  | resnet18 |          256 |          56.26 |           5002.8 |               1294 |            542.3 |
| single_gpu_resnet18_bs256_cacheFalse_ampFalse | resnet18 |          256 |          74    |           4950   |               1294 |            550.4 |
| single_gpu_resnet18_bs512_cacheFalse_ampFalse | resnet18 |          512 |          73.17 |           4893.5 |               2465 |            554.4 |
| model_parallel_resnet18_bs256_cacheFalse      | resnet18 |          256 |          66.74 |           4805.1 |               1423 |            561.9 |
| ddp_nccl_resnet18_bs128x2_cacheTrue_ampFalse  | resnet18 |          256 |          66.71 |           4742.8 |                796 |            724.1 |
| single_gpu_resnet18_bs128_cacheFalse_ampFalse | resnet18 |          128 |          74.77 |           4143.6 |                711 |            675.2 |
| dp_resnet18_bs256_ngpu2_cacheFalse_ampFalse   | resnet18 |          256 |          74.87 |           3805.5 |                715 |            757.3 |
| ddp_nccl_resnet18_bs256x2_cacheFalse_ampFalse | resnet18 |          512 |          73.71 |           2570.5 |               1338 |           1503.1 |
| ddp_nccl_resnet18_bs128x2_cacheFalse_ampTrue  | resnet18 |          256 |          74.45 |           2388.6 |                479 |           1578.3 |
| ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse | resnet18 |          256 |          74.8  |           2293.5 |                753 |           1642.7 |
| collective_nccl_resnet18_bs128x2              | resnet18 |          256 |          67.16 |           2207.6 |                753 |           1705.2 |
| ddp_nccl_resnet34_bs256x2_cacheFalse_ampFalse | resnet34 |          512 |          72.49 |           2095.9 |               2259 |           1750.6 |
| fsdp_grad_resnet18_bs128x2_ampFalse           | resnet18 |          256 |          74.7  |           2014.1 |                709 |           1799.3 |
| fsdp_full_resnet18_bs128x2_ampFalse           | resnet18 |          256 |          74.87 |           1968.2 |                691 |           1827.5 |
| ddp_nccl_resnet18_bs64x2_cacheFalse_ampFalse  | resnet18 |          128 |          74.59 |           1924.5 |                517 |           1893.4 |
| ddp_gloo_resnet18_bs128x2_cacheFalse_ampFalse | resnet18 |          256 |          74.63 |           1833.8 |                758 |           2054.9 |
| ddp_nccl_resnet50_bs256x2_cacheFalse_ampFalse | resnet50 |          512 |          69.67 |           1486.8 |               6584 |           2304.5 |
| collective_gloo_resnet18_bs128x2              | resnet18 |          256 |          66.8  |           1461.5 |                758 |           2298.1 |
| ps_mode_resnet18_bs128_workers2               | resnet18 |          128 |          66.46 |           1209.1 |                669 |           2615.7 |
| ps_gpu_mode_resnet18_bs128_workers2           | resnet18 |          128 |          67.27 |           1067.1 |                669 |           2891.7 |

## 分组实验说明

| 组别 | 实验内容 | 对比项 |
|------|----------|--------|
| Group1 | 单卡 vs 多卡 | Single GPU, DP, DDP |
| Group2 | 通信后端 | NCCL vs Gloo |
| Group3 | 并行策略 | DDP vs ModelParallel vs FSDP |
| Group4 | 缓存效果 | Cache开/关 |
| Group5 | 混合精度 | AMP开/关 |
| Group6 | 批次大小 | BS=64, 128, 256 |
| Group7 | 模型对比 | ResNet18, 34, 50 |
| Group8 | 最优配置 | 基线 vs Cache+AMP |
| Group9 | 批次大小 | BS=128, 256, 512 |
| Group10 | 模式对比 | PS vs. Collective |

## 图表位置

所有图表保存在 `./plots/` 目录下。
