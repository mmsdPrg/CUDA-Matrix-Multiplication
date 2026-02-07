#!/usr/bin/env python3
"""
Profile Analysis Script for nsys and ncu outputs
Extracts and visualizes profiling metrics
"""

import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class ProfileAnalyzer:
    def __init__(self, profile_dir='results/profiles', output_dir='results'):
        self.profile_dir = Path(profile_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = {}
    
    def parse_nsys_stats(self, stats_file):
        """Extract metrics from nsys stats output"""
        with open(stats_file, 'r') as f:
            content = f.read()
        
        metrics = {
            'kernel_time': None,
            'memory_copy_time': None,
            'cuda_api_time': None
        }
        
        # Extract CUDA kernel statistics
        kernel_pattern = r'Time\s*\(%\)\s*Total\s*Time\s*\(ns\)\s*Instances\s*Avg\s*\(ns\)\s*Med\s*\(ns\)\s*Min\s*\(ns\)\s*Max\s*\(ns\)\s*StdDev\s*\(ns\)\s*Name\s*[\d.]+\s*([\d.]+)\s*\d+\s*([\d.]+)'
        
        # Look for matmul_kernel timing
        matmul_pattern = r'matmul_kernel.*?(\d+\.?\d*)\s+(\d+\.?\d*)\s+\d+\s+(\d+\.?\d*)'
        matches = re.findall(matmul_pattern, content)
        if matches:
            # Convert nanoseconds to milliseconds
            metrics['kernel_time'] = float(matches[0][1]) / 1e6
        
        # Extract memory transfer time
        memcpy_pattern = r'\[CUDA memcpy.*?\]\s+[\d.]+\s+([\d.]+)'
        memcpy_matches = re.findall(memcpy_pattern, content)
        if memcpy_matches:
            metrics['memory_copy_time'] = sum(float(m) for m in memcpy_matches) / 1e6
        
        return metrics
    
    def analyze_profiles(self):
        """Analyze all profile stat files"""
        if not self.profile_dir.exists():
            print(f"Profile directory {self.profile_dir} does not exist!")
            return
        
        stat_files = list(self.profile_dir.glob('*_stats.txt'))
        print(f"Found {len(stat_files)} profile stat files")
        
        for stat_file in stat_files:
            profile_name = stat_file.stem.replace('_stats', '')
            print(f"Analyzing profile: {profile_name}...")
            self.metrics[profile_name] = self.parse_nsys_stats(stat_file)
        
        return self.metrics
    
    def create_profile_table(self):
        """Create a table of profiling metrics"""
        if not self.metrics:
            print("No profiling metrics available")
            return
        
        data = []
        for name, metrics in self.metrics.items():
            data.append({
                'Profile': name,
                'Kernel Time (ms)': f"{metrics['kernel_time']:.3f}" if metrics['kernel_time'] else 'N/A',
                'Memory Copy Time (ms)': f"{metrics['memory_copy_time']:.3f}" if metrics['memory_copy_time'] else 'N/A',
                'Total GPU Time (ms)': f"{(metrics['kernel_time'] or 0) + (metrics['memory_copy_time'] or 0):.3f}"
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_file = self.output_dir / 'profile_metrics.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nProfile metrics table saved to {csv_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("PROFILING METRICS TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df
    
    def plot_profile_breakdown(self):
        """Create stacked bar chart showing kernel vs memory time"""
        if not self.metrics:
            return
        
        profiles = []
        kernel_times = []
        memory_times = []
        
        for name, metrics in sorted(self.metrics.items()):
            if metrics['kernel_time'] or metrics['memory_copy_time']:
                profiles.append(name)
                kernel_times.append(metrics['kernel_time'] or 0)
                memory_times.append(metrics['memory_copy_time'] or 0)
        
        if not profiles:
            print("No valid profile data to plot")
            return
        
        x = np.arange(len(profiles))
        width = 0.6
        
        fig, ax = plt.subplots(figsize=(10, 6))
        p1 = ax.bar(x, kernel_times, width, label='Kernel Time', color='steelblue')
        p2 = ax.bar(x, memory_times, width, bottom=kernel_times, label='Memory Copy Time', color='coral')
        
        ax.set_xlabel('Profile', fontweight='bold')
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('GPU Time Breakdown: Kernel vs Memory Transfer', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'profile_breakdown.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Profile breakdown plot saved to {plot_file}")
        plt.close()
    
    def generate_profile_report(self):
        """Generate profiling summary report"""
        if not self.metrics:
            return
        
        report_file = self.output_dir / 'profile_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CUDA PROFILING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("PROFILING METRICS:\n")
            f.write("-" * 80 + "\n")
            
            for name in sorted(self.metrics.keys()):
                metrics = self.metrics[name]
                f.write(f"\n{name}:\n")
                if metrics['kernel_time']:
                    f.write(f"  Kernel Time: {metrics['kernel_time']:.3f} ms\n")
                if metrics['memory_copy_time']:
                    f.write(f"  Memory Copy Time: {metrics['memory_copy_time']:.3f} ms\n")
                
                total_time = (metrics['kernel_time'] or 0) + (metrics['memory_copy_time'] or 0)
                if total_time > 0:
                    f.write(f"  Total GPU Time: {total_time:.3f} ms\n")
                    if metrics['kernel_time']:
                        kernel_pct = (metrics['kernel_time'] / total_time) * 100
                        f.write(f"  Kernel %: {kernel_pct:.1f}%\n")
        
        print(f"Profile report saved to {report_file}")

def main():
    print("="*80)
    print("CUDA Profile Analysis")
    print("="*80 + "\n")
    
    analyzer = ProfileAnalyzer()
    
    # Parse profiles
    analyzer.analyze_profiles()
    
    if not analyzer.metrics:
        print("No profile data found!")
        print("Run 'make profile' first to generate profiling data.")
        return
    
    # Generate outputs
    print("\nGenerating profile analysis...")
    analyzer.create_profile_table()
    analyzer.plot_profile_breakdown()
    analyzer.generate_profile_report()
    
    print("\n" + "="*80)
    print("Profile analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
