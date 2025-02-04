import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

class OlympicInsightsVisualizer:
    def __init__(self, output_dir="outputs/4"):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置颜色方案
        self.colors = {
            'trend': '#2ecc71',      # 绿色
            'region': '#3498db',      # 蓝色
            'country': '#e74c3c',     # 红色
            'neutral': '#95a5a6',     # 灰色
            'highlight': '#f1c40f'    # 黄色
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

    def plot_medal_trends(self, trends: dict, save=True):
        """绘制奖牌趋势分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # 1. 奖牌集中度趋势
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_concentration_trends(ax1, trends.get('overall', {}))

        # 2. 新兴和衰退国家
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_emerging_declining(ax2, trends.get('overall', {}))

        # 3. 区域性趋势
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_regional_trends(ax3, trends.get('regional', {}))

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'medal_trends')
        
        return fig

    def _plot_concentration_trends(self, ax, overall_trends):
        """绘制奖牌集中度趋势"""
        concentration = overall_trends.get('medals_concentration', {})
        
        if not concentration:
            ax.text(0.5, 0.5, 'No concentration data available',
                   ha='center', va='center')
            return
            
        periods = list(concentration.keys())
        gini_coeffs = [d['gini_coefficient'] for d in concentration.values()]
        top10_shares = [d['top_10_share'] for d in concentration.values()]
        
        x = range(len(periods))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], gini_coeffs, width,
               label='Gini Coefficient', color=self.colors['trend'])
        ax.bar([i + width/2 for i in x], top10_shares, width,
               label='Top 10 Share', color=self.colors['highlight'])
        
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.set_title('Medal Concentration Trends')
        ax.legend()

    def _plot_emerging_declining(self, ax, overall_trends):
        """绘制新兴和衰退国家分析"""
        emerging = overall_trends.get('emerging_countries', [])
        declining = overall_trends.get('declining_countries', [])
        
        if not emerging and not declining:
            ax.text(0.5, 0.5, 'No emerging/declining data available',
                   ha='center', va='center')
            return
            
        # 准备数据
        countries = []
        changes = []
        colors = []
        
        for country in emerging[:5]:
            countries.append(country['NOC'])
            changes.append(country['growth_rate'])
            colors.append(self.colors['trend'])
            
        for country in declining[:5]:
            countries.append(country['NOC'])
            changes.append(-country['decline_rate'])
            colors.append(self.colors['country'])
        
        # 绘制水平条形图
        bars = ax.barh(range(len(countries)), changes, color=colors)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.1%}', ha='left' if width > 0 else 'right',
                   va='center')
        
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Emerging vs Declining Countries')

    def _plot_regional_trends(self, ax, regional_trends):
        """绘制区域性趋势分析"""
        if not regional_trends:
            ax.text(0.5, 0.5, 'No regional data available',
                   ha='center', va='center')
            return
            
        regions = list(regional_trends.keys())
        metrics = ['trend', 'volatility', 'medal_median', 'recent_share']
        data = []
        
        for region in regions:
            region_data = regional_trends[region]
            data.append([region_data.get(metric, 0) for metric in metrics])
        
        data = np.array(data)
        
        # 使用平行坐标图
        for i in range(len(regions)):
            ax.plot(range(len(metrics)), data[i, :], 
                   marker='o', label=regions[i])
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_title('Regional Performance Metrics')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_country_insights(self, insights: dict, save=True):
        """绘制国家洞察分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # 1. 综合评分矩阵
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_score_matrix(ax1, insights)

        # 2. 表现稳定性分析
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_stability_analysis(ax2, insights)

        # 3. 多样性分布
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_diversity_distribution(ax3, insights)

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'country_insights')
        
        return fig

    def _plot_score_matrix(self, ax, insights):
        """绘制综合评分矩阵"""
        if not insights:
            ax.text(0.5, 0.5, 'No insights data available',
                   ha='center', va='center')
            return
            
        # 准备数据
        countries = list(insights.keys())[:10]  # 展示前10个国家
        scores = {
            'Trend': [insights[c].trend_score for c in countries],
            'Stability': [insights[c].stability_score for c in countries],
            'Diversity': [insights[c].diversity_score for c in countries]
        }
        
        df = pd.DataFrame(scores, index=countries)
        
        # 绘制热力图
        sns.heatmap(df, cmap='YlOrRd', ax=ax, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Score'})
        
        ax.set_title('Country Performance Matrix')
        ax.set_ylabel('Country')

    def _plot_stability_analysis(self, ax, insights):
        """绘制表现稳定性分析"""
        if not insights:
            ax.text(0.5, 0.5, 'No stability data available',
                   ha='center', va='center')
            return
            
        countries = list(insights.keys())[:10]
        stability_scores = [insights[c].stability_score for c in countries]
        
        # 绘制极坐标图
        angles = np.linspace(0, 2*np.pi, len(countries), endpoint=False)
        
        # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        stability_scores = np.concatenate((stability_scores, [stability_scores[0]]))
        
        ax.plot(angles, stability_scores, 'o-', color=self.colors['trend'])
        ax.fill(angles, stability_scores, alpha=0.25, color=self.colors['trend'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(countries, rotation=45)
        ax.set_title('Performance Stability Analysis')

    def _plot_diversity_distribution(self, ax, insights):
        """绘制多样性分布"""
        if not insights:
            ax.text(0.5, 0.5, 'No diversity data available',
                   ha='center', va='center')
            return
            
        diversity_scores = [insights[c].diversity_score for c in insights.keys()]
        
        sns.histplot(diversity_scores, bins=20, ax=ax, color=self.colors['trend'])
        ax.axvline(np.mean(diversity_scores), color=self.colors['highlight'],
                   linestyle='--', label='Mean')
        
        ax.set_title('Sport Diversity Distribution')
        ax.set_xlabel('Diversity Score')
        ax.set_ylabel('Frequency')
        ax.legend()

    def _save_figure(self, fig, name):
        """保存图表"""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return path