#!/usr/bin/env python3
"""
Analysis Script for CUDA Matrix Multiplication Experiments
Extracts metrics from log files and generates comparison tables/plots
"""

import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class ExperimentAnalyzer:
    def __init__(self, log_dir='results/logs', output_dir='results'):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
        
    def parse_log_file(self, log_file):
        """Extract metrics from a log file"""
        with open(log_file, 'r') as f:
            content = f.read()
        
        metrics = {
            'accuracy': None,
            'avg_loss': None,
            'epoch_times': [],
            'total_time': 0
        }
        
        # Extract accuracy
        acc_pattern = r'Accuracy=(\d+\.\d+)%'
        accuracies = re.findall(acc_pattern, content)
        if accuracies:
            metrics['accuracy'] = float(accuracies[-1])
        
        # Extract loss
        loss_pattern = r'Loss=(\d+\.\d+)'
        losses = re.findall(loss_pattern, content)
        if losses:
            metrics['avg_loss'] = float(losses[-1])
        
        # Extract epoch times
        time_pattern = r'Time=(\d+\.\d+)s'
        times = re.findall(time_pattern, content)
        metrics['epoch_times'] = [float(t) for t in times]
        metrics['total_time'] = sum(metrics['epoch_times'])
        
        return metrics
    
    def analyze_all_logs(self):
        """Parse all log files in the directory"""
        if not self.log_dir.exists():
            print(f"Log directory {self.log_dir} does not exist!")
            return
        
        log_files = list(self.log_dir.glob('*.log'))
        print(f"Found {len(log_files)} log files")
        
        for log_file in log_files:
            experiment_name = log_file.stem
            print(f"Analyzing {experiment_name}...")
            self.results[experiment_name] = self.parse_log_file(log_file)
        
        return self.results
    
    def create_comparison_table(self):
        """Create a comparison table of all experiments"""
        data = []
        for name, metrics in self.results.items():
            data.append({
                'Experiment': name,
                'Accuracy (%)': f"{metrics['accuracy']:.2f}" if metrics['accuracy'] else 'N/A',
                'Avg Loss': f"{metrics['avg_loss']:.4f}" if metrics['avg_loss'] else 'N/A',
                'Total Time (s)': f"{metrics['total_time']:.2f}",
                'Avg Epoch Time (s)': f"{np.mean(metrics['epoch_times']):.2f}" if metrics['epoch_times'] else 'N/A'
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Experiment')
        
        # Save to CSV
        csv_file = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nComparison table saved to {csv_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df
    
    def plot_timing_comparison(self):
        """Create bar plot comparing execution times"""
        experiments = []
        times = []
        
        for name, metrics in sorted(self.results.items()):
            if metrics['total_time'] > 0:
                experiments.append(name)
                times.append(metrics['total_time'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(experiments)), times, color='steelblue', alpha=0.8)
        plt.xlabel('Experiment', fontsize=12, fontweight='bold')
        plt.ylabel('Total Training Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Execution Time Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(experiments)), experiments, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=9)
        
        plot_file = self.output_dir / 'timing_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Timing comparison plot saved to {plot_file}")
        plt.close()
    
    def plot_accuracy_comparison(self):
        """Create bar plot comparing accuracies"""
        experiments = []
        accuracies = []
        
        for name, metrics in sorted(self.results.items()):
            if metrics['accuracy'] is not None:
                experiments.append(name)
                accuracies.append(metrics['accuracy'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(experiments)), accuracies, color='forestgreen', alpha=0.8)
        plt.xlabel('Experiment', fontsize=12, fontweight='bold')
        plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(experiments)), experiments, rotation=45, ha='right')
        plt.ylim([85, 92])  # Expected range 88-91%
        plt.grid(axis='y', alpha=0.3)
        plt.axhline(y=88, color='r', linestyle='--', alpha=0.5, label='Min Expected (88%)')
        plt.axhline(y=91, color='r', linestyle='--', alpha=0.5, label='Max Expected (91%)')
        plt.legend()
        plt.tight_layout()
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plot_file = self.output_dir / 'accuracy_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison plot saved to {plot_file}")
        plt.close()
    
    def plot_category_comparison(self):
        """Create grouped comparison plots by category"""
        categories = {
            'Memory Allocation': ['memory_cudamalloc', 'memory_managed'],
            'Block Size': ['blocksize_8x8', 'blocksize_16x16', 'blocksize_32x32'],
            'Precision': ['precision_fp32', 'precision_fp16', 'precision_int8'],
            'Loop Unrolling': ['unroll_manual', 'unroll_pragma', 'cuda_baseline']
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (category, exp_names) in enumerate(categories.items()):
            ax = axes[idx]
            valid_exps = []
            valid_times = []
            
            for name in exp_names:
                if name in self.results and self.results[name]['total_time'] > 0:
                    valid_exps.append(name.split('_')[-1])
                    valid_times.append(self.results[name]['total_time'])
            
            if valid_exps:
                bars = ax.bar(range(len(valid_exps)), valid_times, color='coral', alpha=0.8)
                ax.set_xlabel('Configuration', fontweight='bold')
                ax.set_ylabel('Time (s)', fontweight='bold')
                ax.set_title(category, fontweight='bold')
                ax.set_xticks(range(len(valid_exps)))
                ax.set_xticklabels(valid_exps, rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, time in zip(bars, valid_times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{time:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'category_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Category comparison plot saved to {plot_file}")
        plt.close()
    
    def generate_report(self):
        """Generate a summary report"""
        report_file = self.output_dir / 'analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CUDA MATRIX MULTIPLICATION OPTIMIZATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Find best performers
            if self.results:
                best_time = min(self.results.items(), 
                               key=lambda x: x[1]['total_time'] if x[1]['total_time'] > 0 else float('inf'))
                best_acc = max(self.results.items(),
                              key=lambda x: x[1]['accuracy'] if x[1]['accuracy'] else 0)
                
                f.write("BEST PERFORMERS:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Fastest: {best_time[0]} ({best_time[1]['total_time']:.2f}s)\n")
                f.write(f"Most Accurate: {best_acc[0]} ({best_acc[1]['accuracy']:.2f}%)\n")
                f.write("\n")
                
                # Baseline comparison
                if 'cuda_baseline' in self.results and 'pytorch_baseline' in self.results:
                    cuda_time = self.results['cuda_baseline']['total_time']
                    pytorch_time = self.results['pytorch_baseline']['total_time']
                    speedup = pytorch_time / cuda_time if cuda_time > 0 else 0
                    
                    f.write("BASELINE COMPARISON:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"PyTorch: {pytorch_time:.2f}s\n")
                    f.write(f"CUDA: {cuda_time:.2f}s\n")
                    f.write(f"Speedup: {speedup:.2f}x\n")
                    f.write("\n")
                
                # Detailed results
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 80 + "\n")
                for name in sorted(self.results.keys()):
                    metrics = self.results[name]
                    f.write(f"\n{name}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.2f}%\n" if metrics['accuracy'] else "  Accuracy: N/A\n")
                    f.write(f"  Avg Loss: {metrics['avg_loss']:.4f}\n" if metrics['avg_loss'] else "  Avg Loss: N/A\n")
                    f.write(f"  Total Time: {metrics['total_time']:.2f}s\n")
                    if metrics['epoch_times']:
                        f.write(f"  Avg Epoch Time: {np.mean(metrics['epoch_times']):.2f}s\n")
        
        print(f"Analysis report saved to {report_file}\n")
        
        
def main():
    print("="*80)
    print("CUDA Matrix Multiplication Analysis")
    print("="*80 + "\n")
    
    analyzer = ExperimentAnalyzer()
    
    # Parse all logs
    analyzer.analyze_all_logs()
    
    if not analyzer.results:
        print("No experiment results found!")
        return
    
    # Generate outputs
    print("\nGenerating analysis outputs...")
    analyzer.create_comparison_table()
    analyzer.plot_timing_comparison()
    analyzer.plot_accuracy_comparison()
    analyzer.plot_category_comparison()
    analyzer.generate_report()
    
    print("\n" + "="*80)
    print("Analysis complete! Check the results/ directory for outputs.")
    print("="*80)

if __name__ == "__main__":
    main()
