"""
实验结果分析和可视化
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_all_results(log_dir='./logs'):
    """加载所有实验结果"""
    results = []
    for filepath in glob.glob(os.path.join(log_dir, '*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
            data['filepath'] = filepath
            results.append(data)
    return results


def create_comparison_table(results):
    """创建对比表格"""
    rows = []
    for r in results:
        row = {
            'Experiment': r.get('experiment_name', 'Unknown'),
            'Training Type': r.get('training_type', 'Unknown'),
            'Model': r.get('model', 'Unknown'),
            'Batch Size': r.get('batch_size', r.get('effective_batch_size', 'N/A')),
            'Best Acc (%)': r.get('best_acc', 0),
            'Total Time (s)': r.get('total_time', 0),
            'Avg Throughput': np.mean(r['metrics']['throughput']) if r.get('metrics') else 0,
            'Max GPU Mem (MB)': max(r['metrics']['gpu_memory']) if r.get('metrics') else 0,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Avg Throughput', ascending=False)
    return df


def plot_training_curves(results, save_dir='./plots'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for r in results:
        name = r.get('experiment_name', 'Unknown')[:30]
        metrics = r.get('metrics', {})
        epochs = range(1, len(metrics.get('train_loss', [])) + 1)
        
        if metrics.get('train_loss'):
            axes[0, 0].plot(epochs, metrics['train_loss'], label=name)
        if metrics.get('test_acc'):
            axes[0, 1].plot(epochs, metrics['test_acc'], label=name)
        if metrics.get('throughput'):
            axes[1, 0].plot(epochs, metrics['throughput'], label=name)
        if metrics.get('gpu_memory'):
            axes[1, 1].plot(epochs, metrics['gpu_memory'], label=name)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Training Throughput')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Samples/sec')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('GPU Memory Usage')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def plot_performance_comparison(results, save_dir='./plots'):
    """绘制性能对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 按训练类型分组
    data = {}
    for r in results:
        training_type = r.get('training_type', 'Unknown')
        if training_type not in data:
            data[training_type] = {
                'throughput': [],
                'best_acc': [],
                'gpu_memory': [],
                'total_time': []
            }
        
        metrics = r.get('metrics', {})
        data[training_type]['throughput'].append(np.mean(metrics.get('throughput', [0])))
        data[training_type]['best_acc'].append(r.get('best_acc', 0))
        data[training_type]['gpu_memory'].append(max(metrics.get('gpu_memory', [0])))
        data[training_type]['total_time'].append(r.get('total_time', 0))
    
    # 创建对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    types = list(data.keys())
    x = np.arange(len(types))
    width = 0.6
    
    # 吞吐量对比
    throughputs = [np.mean(data[t]['throughput']) for t in types]
    axes[0, 0].bar(x, throughputs, width, color='steelblue')
    axes[0, 0].set_title('Average Throughput Comparison')
    axes[0, 0].set_ylabel('Samples/sec')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(types, rotation=45, ha='right')
    axes[0, 0].grid(axis='y')
    
    # 准确率对比
    accs = [np.mean(data[t]['best_acc']) for t in types]
    axes[0, 1].bar(x, accs, width, color='forestgreen')
    axes[0, 1].set_title('Best Accuracy Comparison')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(types, rotation=45, ha='right')
    axes[0, 1].grid(axis='y')
    
    # GPU内存对比
    mems = [np.mean(data[t]['gpu_memory']) for t in types]
    axes[1, 0].bar(x, mems, width, color='coral')
    axes[1, 0].set_title('Max GPU Memory Comparison')
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(types, rotation=45, ha='right')
    axes[1, 0].grid(axis='y')
    
    # 总训练时间对比
    times = [np.mean(data[t]['total_time']) for t in types]
    axes[1, 1].bar(x, times, width, color='mediumpurple')
    axes[1, 1].set_title('Total Training Time Comparison')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(types, rotation=45, ha='right')
    axes[1, 1].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=150)
    plt.close()
    print(f"Performance comparison saved to {save_dir}/performance_comparison.png")


def plot_scalability_analysis(results, save_dir='./plots'):
    """绘制扩展性分析图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 分析不同GPU数量/batch size的扩展性
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 按world_size分组
    ddp_results = [r for r in results if 'ddp' in r.get('training_type', '').lower()]
    
    if ddp_results:
        world_sizes = []
        throughputs = []
        for r in ddp_results:
            ws = r.get('world_size', 1)
            tp = np.mean(r.get('metrics', {}).get('throughput', [0]))
            world_sizes.append(ws)
            throughputs.append(tp)
        
        if world_sizes:
            axes[0].plot(world_sizes, throughputs, 'o-', markersize=10, linewidth=2)
            # 理想线性扩展
            if throughputs:
                ideal = [throughputs[0] * ws / world_sizes[0] for ws in world_sizes]
                axes[0].plot(world_sizes, ideal, '--', color='gray', label='Ideal Linear')
            axes[0].set_title('DDP Scalability (Throughput vs GPUs)')
            axes[0].set_xlabel('Number of GPUs')
            axes[0].set_ylabel('Throughput (samples/sec)')
            axes[0].legend()
            axes[0].grid(True)
    
    # Batch size vs Throughput
    batch_sizes = []
    throughputs = []
    for r in results:
        bs = r.get('batch_size', r.get('effective_batch_size', 0))
        tp = np.mean(r.get('metrics', {}).get('throughput', [0]))
        if bs and tp:
            batch_sizes.append(bs)
            throughputs.append(tp)
    
    if batch_sizes:
        axes[1].scatter(batch_sizes, throughputs, s=100, alpha=0.7)
        axes[1].set_title('Batch Size vs Throughput')
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Throughput (samples/sec)')
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scalability_analysis.png'), dpi=150)
    plt.close()
    print(f"Scalability analysis saved to {save_dir}/scalability_analysis.png")


def generate_report(results, output_file='./experiment_report.md'):
    """生成实验报告"""
    df = create_comparison_table(results)
    
    report = """# 分布式训练实验报告

## 实验概述

本实验在CIFAR-100数据集上对比了多种分布式训练策略的性能。

## 实验配置

- 数据集: CIFAR-100 (fine_label, 100类)
- 硬件: 2x NVIDIA A40 (48GB)
- 基础模型: ResNet50

## 实验结果汇总

"""
    report += df.to_markdown(index=False)
    
    report += """

## 详细分析

### 1. 吞吐量分析
- 最高吞吐量配置: {}
- 最低吞吐量配置: {}

### 2. 准确率分析
- 最高准确率配置: {}
- 最低准确率配置: {}

### 3. 内存效率分析
- 内存使用最少配置: {}
- 内存使用最多配置: {}

## 结论与建议

基于实验结果，建议:
1. 对于追求训练速度的场景，使用DDP数据并行
2. 对于内存受限的场景，考虑FSDP或模型并行
3. 混合精度训练(AMP)可以显著提升吞吐量并减少内存使用

""".format(
        df.iloc[0]['Experiment'] if len(df) > 0 else 'N/A',
        df.iloc[-1]['Experiment'] if len(df) > 0 else 'N/A',
        df.sort_values('Best Acc (%)', ascending=False).iloc[0]['Experiment'] if len(df) > 0 else 'N/A',
        df.sort_values('Best Acc (%)', ascending=True).iloc[0]['Experiment'] if len(df) > 0 else 'N/A',
        df.sort_values('Max GPU Mem (MB)', ascending=True).iloc[0]['Experiment'] if len(df) > 0 else 'N/A',
        df.sort_values('Max GPU Mem (MB)', ascending=False).iloc[0]['Experiment'] if len(df) > 0 else 'N/A',
    )
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_file}")
    return report


def main():
    """主函数"""
    print("Loading experiment results...")
    results = load_all_results()
    
    if not results:
        print("No results found in ./logs directory")
        return
    
    print(f"Found {len(results)} experiments")
    
    # 创建对比表格
    df = create_comparison_table(results)
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # 绘制图表
    print("\nGenerating plots...")
    plot_training_curves(results)
    plot_performance_comparison(results)
    plot_scalability_analysis(results)
    
    # 生成报告
    print("\nGenerating report...")
    generate_report(results)


if __name__ == '__main__':
    main()