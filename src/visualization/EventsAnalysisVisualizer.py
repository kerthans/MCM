import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec
from typing import Dict, List

class EventsAnalysisVisualizer:
    def __init__(self, output_dir="outputs/2"):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置颜色方案
        self.colors = {
            'correlation': '#2ecc71',
            'sports': '#3498db',
            'host': '#e74c3c',
            'neutral': '#95a5a6'
        }
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        self._setup_style()
    
    def _setup_style(self):
        """设置matplotlib样式"""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })

    def plot_events_correlation(self, correlation_results: Dict, save=True):
        """绘制赛事相关性分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # 1. 整体相关性趋势
        ax1 = fig.add_subplot(gs[0, :])
        overall_corr = correlation_results.get('overall_correlation', 0)
        ax1.bar(['Overall Correlation'], [overall_corr], 
                color=self.colors['correlation'], alpha=0.7)
        ax1.set_ylim(-1, 1)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title('Overall Events-Medals Correlation')
        
        # 2. 各国相关性对比
        ax2 = fig.add_subplot(gs[1, :])
        country_corrs = correlation_results.get('country_correlations', {})
        countries = list(country_corrs.keys())
        correlations = list(country_corrs.values())
        
        bars = ax2.barh(countries, correlations, 
                     color=self.colors['correlation'], alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        ax2.set_title('Country-Specific Events-Medals Correlation')
        ax2.set_xlabel('Correlation Coefficient')

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'events_correlation')
        
        return fig

    def plot_key_sports_analysis(self, sports_strength: Dict, top_n=5, save=True):
        """绘制重点运动项目分析图"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2)

        # 1. 顶级国家的项目分布
        ax1 = fig.add_subplot(gs[0, :])
        top_countries = list(sports_strength.keys())[:top_n]
        
        data = []
        for country in top_countries:
            for sport in sports_strength[country]:
                data.append({
                    'Country': country,
                    'Sport': sport['sport'],
                    'Medal_Count': sport['medal_count'],
                    'Percentage': sport['percentage']
                })
        
        df = pd.DataFrame(data)
        pivot_table = df.pivot(index='Sport', columns='Country', values='Medal_Count')
        
        sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax1, 
                    annot=True, fmt='.0f', cbar_kws={'label': 'Medal Count'})
        ax1.set_title('Key Sports Distribution by Country')

        # 2. 项目专注度分析
        ax2 = fig.add_subplot(gs[1, 0])
        
        for country in top_countries:
            percentages = [sport['percentage'] for sport in sports_strength[country]]
            ax2.plot(range(1, len(percentages) + 1), percentages, 
                    marker='o', label=country)
        
        ax2.set_xlabel('Sport Rank')
        ax2.set_ylabel('Medal Percentage')
        ax2.set_title('Sport Concentration Analysis')
        ax2.legend()

        # 3. 全球排名分布
        ax3 = fig.add_subplot(gs[1, 1])
        
        for country in top_countries:
            ranks = [sport['global_rank'] for sport in sports_strength[country]]
            ax3.plot(range(1, len(ranks) + 1), ranks, 
                    marker='o', label=country)
        
        ax3.set_xlabel('Sport Rank')
        ax3.set_ylabel('Global Ranking')
        ax3.set_title('Global Ranking Distribution')
        ax3.invert_yaxis()  # 让排名1在顶部
        ax3.legend()

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'key_sports_analysis')
        
        return fig

    def plot_host_impact(self, host_impact: Dict, save=True):
        """绘制主办国效应分析图"""
        host_effects = host_impact.get('host_effects', pd.DataFrame())
        if host_effects.empty:
            return None

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # 1. 主办年份效应对比
        ax1 = fig.add_subplot(gs[0, :])
        
        impact_data = host_effects.sort_values('Year')
        ax1.plot(impact_data['Year'], impact_data['Impact'], 
                marker='o', color=self.colors['host'])
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title('Host Country Impact Over Time')
        ax1.set_xlabel('Olympic Year')
        ax1.set_ylabel('Medal Impact')

        # 2. 前后表现对比
        ax2 = fig.add_subplot(gs[1, 0])
        
        performance_data = pd.DataFrame({
            'Phase': ['Before', 'During', 'After'],
            'Average': [
                host_effects['Before'].mean(),
                host_effects['During'].mean(),
                host_effects['After'].mean()
            ]
        })
        
        bars = ax2.bar(performance_data['Phase'], performance_data['Average'],
                      color=self.colors['host'], alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        ax2.set_title('Average Performance by Phase')
        ax2.set_ylabel('Average Medals')

        # 3. 最近主办国分析
        ax3 = fig.add_subplot(gs[1, 1])
        
        recent_hosts = host_effects.nlargest(5, 'Impact')
        bars = ax3.barh(recent_hosts['Host'], recent_hosts['Impact'],
                       color=self.colors['host'], alpha=0.7)
        
        for bar in bars:
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}', ha='left', va='center')
        
        ax3.set_title('Top 5 Host Country Impacts')
        ax3.set_xlabel('Medal Impact')

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'host_impact_analysis')
        
        return fig

    def _save_figure(self, fig, name):
        """保存图表"""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return path