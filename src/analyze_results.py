"""
实验结果分析和可视化
按实验组分组绘制图表
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_all_results(log_dir='./logs'):
    """加载所有实验结果"""
    results = []
    for filepath in glob.glob(os.path.join(log_dir, '*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
            data['filepath'] = filepath
            results.append(data)
    return results


def get_short_name(result):
    """获取简短的实验名称用于图例"""
    exp_name = result.get('experiment_name', 'Unknown')
    
    # 确定训练类型
    if exp_name.startswith('single_gpu'):
        name = 'SingleGPU'
    elif exp_name.startswith('dp_'):
        name = 'DP'
    elif exp_name.startswith('ddp_nccl'):
        name = 'DDP-NCCL'
    elif exp_name.startswith('ddp_gloo'):
        name = 'DDP-Gloo'
    elif exp_name.startswith('fsdp_full'):
        name = 'FSDP-Full'
    elif exp_name.startswith('fsdp_grad'):
        name = 'FSDP-Grad'
    elif exp_name.startswith('model_parallel'):
        name = 'ModelParallel'
    else:
        name = exp_name[:15]
    
    # 提取模型名称
    model = result.get('model', '')
    if model:
        name += f'-{model}'
    
    # 提取batch size
    bs = result.get('batch_size', result.get('batch_size_per_gpu', ''))
    if bs:
        name += f'-bs{bs}'
    
    # 添加cache和amp标记
    if 'cacheTrue' in exp_name:
        name += '+Cache'
    if 'ampTrue' in exp_name:
        name += '+AMP'
    
    return name


def plot_group_figure(results, group_name, save_dir='./plots'):
    """
    绘制单个实验组的图表（4个子图）
    """
    if not results:
        print(f"No results for group: {group_name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{group_name}', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    
    for idx, r in enumerate(results):
        name = get_short_name(r)
        metrics = r.get('metrics', {})
        epochs = range(1, len(metrics.get('train_loss', [])) + 1)
        color = colors[idx]
        
        if metrics.get('train_loss'):
            axes[0, 0].plot(epochs, metrics['train_loss'], label=name, color=color, linewidth=1.5)
        if metrics.get('test_acc'):
            axes[0, 1].plot(epochs, metrics['test_acc'], label=name, color=color, linewidth=1.5)
        if metrics.get('throughput'):
            axes[1, 0].plot(epochs, metrics['throughput'], label=name, color=color, linewidth=1.5)
        if metrics.get('gpu_memory'):
            axes[1, 1].plot(epochs, metrics['gpu_memory'], label=name, color=color, linewidth=1.5)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='upper right', fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(loc='lower right', fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Throughput')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Samples/sec')
    axes[1, 0].legend(loc='lower right', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('GPU Memory Usage')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].legend(loc='upper right', fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_name = group_name.replace(' ', '_').replace(':', '').replace('/', '_').replace('-', '_')
    filepath = os.path.join(save_dir, f'{safe_name}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_group1_single_vs_multi(results, save_dir='./plots'):
    """
    实验组1: 单卡 vs 多卡对比
    - single_gpu_resnet18_bs128_cacheFalse_ampFalse
    - dp_resnet18_bs256_ngpu2_cacheFalse_ampFalse  
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse
    """
    group_results = []
    target_names = [
        'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
        'dp_resnet18_bs256_ngpu2_cacheFalse_ampFalse',
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group1_Single_vs_Multi_GPU', save_dir)


def plot_group2_backend_comparison(results, save_dir='./plots'):
    """
    实验组2: 通信后端对比 (NCCL vs Gloo)
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse
    - ddp_gloo_resnet18_bs128x2_cacheFalse_ampFalse
    """
    group_results = []
    target_names = [
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        'ddp_gloo_resnet18_bs128x2_cacheFalse_ampFalse',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group2_Backend_NCCL_vs_Gloo', save_dir)


def plot_group3_parallel_strategies(results, save_dir='./plots'):
    """
    实验组3: 并行策略对比
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse (数据并行)
    - model_parallel_resnet18_bs256_cacheFalse (模型并行)
    - fsdp_full_resnet18_bs128x2_ampFalse (FSDP Full)
    - fsdp_grad_resnet18_bs128x2_ampFalse (FSDP Grad)
    """
    group_results = []
    target_names = [
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        'model_parallel_resnet18_bs256_cacheFalse',
        'fsdp_full_resnet18_bs128x2_ampFalse',
        'fsdp_grad_resnet18_bs128x2_ampFalse',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group3_Parallel_Strategies', save_dir)


def plot_group4_cache_comparison(results, save_dir='./plots'):
    """
    实验组4: 缓存对比
    - single_gpu_resnet18_bs256_cacheFalse_ampFalse
    - single_gpu_resnet18_bs256_cacheTrue_ampFalse
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse
    - ddp_nccl_resnet18_bs128x2_cacheTrue_ampFalse
    """
    group_results = []
    target_names = [
        'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
        'single_gpu_resnet18_bs256_cacheTrue_ampFalse',
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet18_bs128x2_cacheTrue_ampFalse',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group4_Cache_Comparison', save_dir)


def plot_group5_amp_comparison(results, save_dir='./plots'):
    """
    实验组5: 混合精度(AMP)对比
    - single_gpu_resnet18_bs256_cacheFalse_ampFalse
    - single_gpu_resnet18_bs256_cacheFalse_ampTrue
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampTrue
    """
    group_results = []
    target_names = [
        'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
        'single_gpu_resnet18_bs256_cacheFalse_ampTrue',
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampTrue',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group5_AMP_Comparison', save_dir)


def plot_group6_batch_size(results, save_dir='./plots'):
    """
    实验组6: Batch Size对比
    - ddp_nccl_resnet18_bs64x2_cacheFalse_ampFalse
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse
    - ddp_nccl_resnet18_bs256x2_cacheFalse_ampFalse
    """
    group_results = []
    target_names = [
        'ddp_nccl_resnet18_bs64x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet18_bs256x2_cacheFalse_ampFalse',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group6_Batch_Size', save_dir)


def plot_group7_model_comparison(results, save_dir='./plots'):
    """
    实验组7: 不同模型对比
    - ddp_nccl_resnet18_bs256x2_cacheFalse_ampFalse
    - ddp_nccl_resnet34_bs256x2_cacheFalse_ampFalse
    - ddp_nccl_resnet50_bs256x2_cacheFalse_ampFalse
    """
    group_results = []
    target_names = [
        'ddp_nccl_resnet18_bs256x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet34_bs256x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet50_bs256x2_cacheFalse_ampFalse',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group7_Model_Comparison', save_dir)


def plot_group8_optimal_config(results, save_dir='./plots'):
    """
    实验组8: 最优配置对比 (Cache + AMP)
    - single_gpu_resnet18_bs256_cacheFalse_ampFalse (基线)
    - single_gpu_resnet18_bs256_cacheTrue_ampTrue (单卡最优)
    - ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse (DDP基线)
    - ddp_nccl_resnet18_bs128x2_cacheTrue_ampTrue (DDP最优)
    """
    group_results = []
    target_names = [
        'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
        'single_gpu_resnet18_bs256_cacheTrue_ampTrue',
        'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        'ddp_nccl_resnet18_bs128x2_cacheTrue_ampTrue',
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group8_Optimal_Config', save_dir)

def plot_group9_single_bsz(results, save_dir='./plots'):
    """
    实验组9: single gpu bsz 对比
    - single_gpu_resnet18_bs128_cacheFalse_ampFalse
    - single_gpu_resnet18_bs256_cacheFalse_ampFalse
    - single_gpu_resnet18_bs512_cacheFalse_ampFalse
    """
    group_results = []
    target_names = [
        'single_gpu_resnet18_bs128_cacheFalse_ampFalse',
        'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
        'single_gpu_resnet18_bs512_cacheFalse_ampFalse'
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group9_singlegpu_bsz', save_dir)

def plot_group10_ps(results, save_dir='./plots'):

    group_results = []
    target_names = [
        'collective_gloo_resnet18_bs128x2',
        'collective_nccl_resnet18_bs128x2',
        'ps_mode_resnet18_bs128_workers2',
        'ps_gpu_mode_resnet18_bs128_workers2'
    ]
    
    for r in results:
        exp_name = r.get('experiment_name', '')
        if exp_name in target_names:
            group_results.append(r)
    
    plot_group_figure(group_results, 'Group10_ps_collective', save_dir)


def plot_summary_bar_chart(results, save_dir='./plots'):
    """绘制总体性能对比柱状图"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Overall Performance Summary', fontsize=14, fontweight='bold')
    
    names = []
    throughputs = []
    accuracies = []
    memories = []
    times = []
    
    for r in results:
        name = get_short_name(r)
        metrics = r.get('metrics', {})
        
        names.append(name)
        throughputs.append(np.mean(metrics.get('throughput', [0])))
        accuracies.append(r.get('best_acc', 0))
        memories.append(max(metrics.get('gpu_memory', [0])) if metrics.get('gpu_memory') else 0)
        times.append(r.get('total_time', 0))
    
    x = np.arange(len(names))
    width = 0.6
    
    axes[0, 0].bar(x, throughputs, width, color='steelblue')
    axes[0, 0].set_title('Average Throughput')
    axes[0, 0].set_ylabel('Samples/sec')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    axes[0, 1].bar(x, accuracies, width, color='forestgreen')
    axes[0, 1].set_title('Best Test Accuracy')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    axes[1, 0].bar(x, memories, width, color='coral')
    axes[1, 0].set_title('Peak GPU Memory')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    axes[1, 1].bar(x, times, width, color='mediumpurple')
    axes[1, 1].set_title('Total Training Time')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=6)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'Summary_All_Experiments.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def create_comparison_table(results):
    """创建对比表格"""
    rows = []
    for r in results:
        metrics = r.get('metrics', {})
        row = {
            'Experiment': r.get('experiment_name', 'Unknown'),
            'Model': r.get('model', 'N/A'),
            'Batch Size': r.get('batch_size', r.get('effective_batch_size', r.get('batch_size_per_gpu', 'N/A'))),
            'Best Acc (%)': f"{r.get('best_acc', 0):.2f}",
            'Avg Throughput': f"{np.mean(metrics.get('throughput', [0])):.1f}",
            'Max GPU Mem (MB)': f"{max(metrics.get('gpu_memory', [0])):.0f}" if metrics.get('gpu_memory') else 'N/A',
            'Total Time (s)': f"{r.get('total_time', 0):.1f}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Avg Throughput', ascending=False)
    return df


def get_group_results(results, target_names):
    """根据目标名称列表筛选结果"""
    return [r for r in results if r.get('experiment_name', '') in target_names]


def print_grouped_tables(results):
    """按实验组分组打印表格"""
    
    # 定义各组实验
    groups = {
        'Group1: Single vs Multi-GPU': [
            'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
            'dp_resnet18_bs256_ngpu2_cacheFalse_ampFalse',
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
        ],
        'Group2: Backend (NCCL vs Gloo)': [
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'ddp_gloo_resnet18_bs128x2_cacheFalse_ampFalse',
        ],
        'Group3: Parallel Strategies': [
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'model_parallel_resnet18_bs256_cacheFalse',
            'fsdp_full_resnet18_bs128x2_ampFalse',
            'fsdp_grad_resnet18_bs128x2_ampFalse',
        ],
        'Group4: Cache Comparison': [
            'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
            'single_gpu_resnet18_bs256_cacheTrue_ampFalse',
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet18_bs128x2_cacheTrue_ampFalse',
        ],
        'Group5: AMP Comparison': [
            'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
            'single_gpu_resnet18_bs256_cacheFalse_ampTrue',
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampTrue',
        ],
        'Group6: Batch Size': [
            'ddp_nccl_resnet18_bs64x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet18_bs256x2_cacheFalse_ampFalse',
        ],
        'Group7: Model Comparison': [
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet34_bs128x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet50_bs128x2_cacheFalse_ampFalse',
        ],
        'Group8: Optimal Config (Cache+AMP)': [
            'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
            'single_gpu_resnet18_bs256_cacheTrue_ampTrue',
            'ddp_nccl_resnet18_bs128x2_cacheFalse_ampFalse',
            'ddp_nccl_resnet18_bs128x2_cacheTrue_ampTrue',
        ],
        'Group9: single gpu bsz': [
            'single_gpu_resnet18_bs128_cacheFalse_ampFalse',
            'single_gpu_resnet18_bs256_cacheFalse_ampFalse',
            'single_gpu_resnet18_bs512_cacheFalse_ampFalse'
        ],
        'Group10: PS vs. Collective': [
            'collective_gloo_resnet18_bs128x2',
            'collective_nccl_resnet18_bs128x2',
            'ps_mode_resnet18_bs128_workers2',
            'ps_gpu_mode_resnet18_bs128_workers2'
        ]
    }
    
    for group_name, target_names in groups.items():
        group_results = get_group_results(results, target_names)
        
        if not group_results:
            print(f"\n{'='*60}")
            print(f"{group_name}")
            print(f"{'='*60}")
            print("  (No matching experiments found)")
            continue
        
        df = create_comparison_table(group_results)
        
        print(f"\n{'='*80}")
        print(f"{group_name}")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # 计算并打印关键对比指标
        if len(group_results) >= 2:
            throughputs = [float(row['Avg Throughput']) for _, row in df.iterrows()]
            accs = [float(row['Best Acc (%)']) for _, row in df.iterrows()]
            
            max_tp = max(throughputs)
            min_tp = min(throughputs)
            if min_tp > 0:
                speedup = max_tp / min_tp
                print(f"\n  → Throughput range: {min_tp:.1f} ~ {max_tp:.1f} samples/s (speedup: {speedup:.2f}x)")
            
            print(f"  → Accuracy range: {min(accs):.2f}% ~ {max(accs):.2f}%")


def generate_report(results, output_file='./experiment_report.md'):
    """生成实验报告"""
    df = create_comparison_table(results)
    
    report = """# 分布式训练实验报告

## 实验结果汇总

"""
    report += df.to_markdown(index=False)
    
    report += """

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
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_file}")


def main():
    """主函数"""
    log_dir = os.environ.get('LOG_DIR', './logs')
    save_dir = './plots'
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading experiment results from {log_dir}...")
    results = load_all_results(log_dir)
    
    if not results:
        print(f"No results found in {log_dir} directory")
        return
    
    print(f"Found {len(results)} experiments:")
    for r in results:
        print(f"  - {r.get('experiment_name', 'Unknown')}")
    
    # 打印总表
    df = create_comparison_table(results)
    print("\n" + "="*80)
    print("ALL EXPERIMENTS (sorted by throughput)")
    print("="*80)
    print(df.to_string(index=False))
    
    # 按组打印对比表格
    print_grouped_tables(results)
    
    # 按实验组绘制图表
    print("\n" + "="*80)
    print("Generating group plots...")
    print("="*80)
    
    plot_group1_single_vs_multi(results, save_dir)
    plot_group2_backend_comparison(results, save_dir)
    plot_group3_parallel_strategies(results, save_dir)
    plot_group4_cache_comparison(results, save_dir)
    plot_group5_amp_comparison(results, save_dir)
    plot_group6_batch_size(results, save_dir)
    plot_group7_model_comparison(results, save_dir)
    plot_group8_optimal_config(results, save_dir)
    plot_group9_single_bsz(results, save_dir)
    plot_group10_ps(results, save_dir)
    
    # 总体对比图
    print("\nGenerating summary plot...")
    plot_summary_bar_chart(results, save_dir)
    
    # 生成报告
    print("\nGenerating report...")
    generate_report(results)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Plots saved to: {save_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()