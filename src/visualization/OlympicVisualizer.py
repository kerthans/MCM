import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

class OlympicVisualizer:
    def __init__(self, output_dir="outputs/1"):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置颜色方案
        self.colors = {
            'gold': '#FFD700',
            'silver': '#C0C0C0',
            'bronze': '#CD7F32',
            'primary': '#2E86C1',
            'secondary': '#E74C3C',
            'accent': '#2ECC71',
            'neutral': '#95A5A6'
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

    def plot_prediction_results(self, predictions_df, save=True):
        """优化的预测结果综合图表"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, height_ratios=[2, 1])

        # 1. 金牌预测条形图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_medal_predictions(ax1, predictions_df, 'Gold', 
                                '2028 Gold Medal Predictions')

        # 2. 总奖牌预测条形图
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_medal_predictions(ax2, predictions_df, 'Total', 
                                '2028 Total Medal Predictions')

        # 3. 不确定性分析
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_uncertainty_analysis(ax3, predictions_df)

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'prediction_results')
        
        return fig

    def _plot_medal_predictions(self, ax, df, medal_type, title):
        """优化的奖牌预测条形图"""
        # 颜色映射
        tier_colors = {
            1: '#2ecc71',  # 绿色
            2: '#3498db',  # 蓝色
            3: '#95a5a6'   # 灰色
        }
        
        top_10 = df.nlargest(10, f'Predicted_{medal_type}')
        
        # 根据tier分配颜色
        colors = [tier_colors[tier] for tier in top_10['country_tier']]
        
        # 绘制条形图
        bars = ax.barh(range(len(top_10)), 
                    top_10[f'Predicted_{medal_type}'],
                    color=colors, alpha=0.7)
        
        # 添加误差条
        ax.errorbar(top_10[f'Predicted_{medal_type}'], 
                range(len(top_10)),
                xerr=top_10[f'{medal_type}_Uncertainty'],
                fmt='none', color='black', capsize=5)
        
        # 设置标签等
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['Country'])
        ax.set_title(title)
        ax.set_xlabel('Predicted Medals')
        
        # 添加分层图例
        legend_elements = [Patch(facecolor=color, alpha=0.7, label=f'Tier {tier}')
                        for tier, color in tier_colors.items()]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1))

    def _plot_uncertainty_analysis(self, ax, df):
        """优化的不确定性分析图"""
        uncertainty_data = pd.DataFrame({
            'Gold': df['Gold_Uncertainty'],
            'Total': df['Total_Uncertainty']
        })
        
        # 添加层级信息
        tiers = pd.Categorical(df['country_tier'])
        colors = ['#2ecc71', '#3498db', '#95a5a6']
        
        sns.boxplot(data=uncertainty_data, ax=ax,
                palette=['#f1c40f', '#3498db'])
        
        # 添加离群点的形状和颜色映射到层级
        for i, column in enumerate(uncertainty_data.columns):
            scatter_data = uncertainty_data[column]
            q1, q3 = scatter_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = scatter_data[(scatter_data < q1 - 1.5 * iqr) | 
                                (scatter_data > q3 + 1.5 * iqr)]
            if not outliers.empty:
                ax.scatter([i] * len(outliers), outliers,
                        c=[colors[t-1] for t in df.loc[outliers.index, 'country_tier']],
                        marker='o', alpha=0.6)
        
        ax.set_title('Prediction Uncertainty Analysis')
        ax.set_ylabel('Uncertainty Range (medals)')
        ax.set_xticklabels(['Gold Medals', 'Total Medals'])

    def plot_historical_comparison(self, historical_data, predictions_df, host_data=None):
        """优化的历史对比分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1])

        # 1. 历史趋势图
        ax1 = fig.add_subplot(gs[0])
        self._plot_historical_trends(ax1, historical_data, host_data)

        # 2. 2024-2028对比图
        ax2 = fig.add_subplot(gs[1])
        self._plot_olympics_comparison(ax2, predictions_df)

        plt.tight_layout()
        self._save_figure(fig, 'historical_comparison')
        
        return fig

    def _plot_historical_trends(self, ax, historical_data, host_data=None):
        """优化的历史趋势线图"""
        # 处理重复项的数据准备
        top_countries = historical_data.groupby('NOC')['Total'].mean().nlargest(5).index
        
        # 颜色映射
        color_map = {
            'United States': '#1f77b4',
            'China': '#ff7f0e',
            'Great Britain': '#2ca02c',
            'Soviet Union': '#d62728',
            'Australia': '#9467bd'
        }
        
        # 绘制主要趋势线
        for country in top_countries:
            country_data = historical_data[historical_data['NOC'] == country]
            ax.plot(country_data['Year'], country_data['Total'], 
                marker='o', label=country, color=color_map.get(country),
                linewidth=2, markersize=6)
        
        # 添加主办国标记
        if host_data is not None:
            usa_hosts = host_data[host_data['NOC'] == 'United States']['Year'].unique()
            for year in usa_hosts:
                ax.axvline(x=year, color='gray', alpha=0.3, linestyle='--')
                ax.fill_between([year-0.5, year+0.5], 0, ax.get_ylim()[1],
                            color='gray', alpha=0.1)
        
        # 移除重复的图例项
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        ax.legend(unique_handles, unique_labels)
        ax.set_title('Historical Medal Trends (Top 5 Countries)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Medals')
        ax.grid(True, alpha=0.3)

    def _plot_olympics_comparison(self, ax, df):
        """优化的奥运会对比图"""
        top_10 = df.nlargest(10, 'Predicted_Total')
        
        x = np.arange(len(top_10))
        width = 0.35
        
        # 计算变化百分比
        pct_change = ((top_10['Predicted_Total'] - top_10['2024_Total']) / 
                    top_10['2024_Total'] * 100)
        
        # 绘制条形图
        bars1 = ax.bar(x - width/2, top_10['2024_Total'], width,
                    label='2024 Actual', color='#b3b3b3', alpha=0.7)
        bars2 = ax.bar(x + width/2, top_10['Predicted_Total'], width,
                    label='2028 Predicted', color='#2ecc71', alpha=0.7)
        
        # 添加百分比变化标签
        for i, (pct, bar) in enumerate(zip(pct_change, bars2)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f'{pct:+.1f}%',
                    ha='center', va='bottom', rotation=0,
                    fontsize=8, color='#2c3e50')
        
        ax.set_title('2024 vs 2028 Medal Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(top_10['Country'], rotation=45, ha='right')
        ax.set_ylabel('Total Medals')
        
        # 添加图例
        legend = ax.legend(title='comparison', bbox_to_anchor=(1.05, 1))
        legend.get_title().set_fontweight('bold')

    def plot_first_time_medalists(self, first_time_data, save=True):
        """绘制首次获奖国家分析图"""
        if not first_time_data:
            return None
            
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 2)

        # 1. 预测奖牌数
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_first_time_medals(ax1, first_time_data)

        # 2. 获奖概率分析
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_first_time_probability(ax2, first_time_data)

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'first_time_medalists')
        
        return fig

    def _plot_first_time_medals(self, ax, data):
        """绘制首次获奖奖牌预测图"""
        countries = list(data.keys())
        predictions = [info['predicted_medals'] for info in data.values()]
        uncertainties = [info['uncertainty'] for info in data.values()]
        
        bars = ax.barh(range(len(countries)), predictions,
                      color=self.colors['accent'], alpha=0.7)
        
        ax.errorbar(predictions, range(len(countries)),
                   xerr=uncertainties, fmt='none', color='black', capsize=5)
        
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries)
        ax.set_title('Predicted Medals for First-Time Medalists')
        ax.set_xlabel('Predicted Medals')

    def _plot_first_time_probability(self, ax, data):
        """绘制首次获奖概率分析图"""
        probabilities = [info['probability'] for info in data.values()]
        
        sns.histplot(probabilities, bins=10, ax=ax, color=self.colors['accent'])
        ax.axvline(np.mean(probabilities), color=self.colors['secondary'],
                   linestyle='--', label='Mean Probability')
        
        ax.set_title('Winning Probability Distribution')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Frequency')
        ax.legend()

    def _save_figure(self, fig, name):
        """保存图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{name}_{timestamp}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return path
# 在 OlympicVisualizer 类中添加的新方法
    def plot_r2_scores(self, r2_scores_data, save=True):
        """
        绘制模型R²Score的拟合线型图
        
        Parameters:
        -----------
        r2_scores_data : dict
            包含每个模型在不同目标上的R²分数
            格式: {
                'Gold': {'gbm': [...], 'rf': [...], 'xgb': [...], 'lgb': [...]},
                'Total': {'gbm': [...], 'rf': [...], 'xgb': [...], 'lgb': [...]}
            }
        save : bool
            是否保存图表
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 颜色映射
        model_colors = {
            'gbm': '#2ecc71',  # 绿色
            'rf': '#3498db',   # 蓝色
            'xgb': '#e74c3c',  # 红色
            'lgb': '#f1c40f'   # 黄色
        }
        
        # 为每个目标（Gold和Total）创建子图
        for idx, target in enumerate(['Gold', 'Total']):
            ax = ax1 if idx == 0 else ax2
            
            # 绘制每个模型的R²分数
            for model_name, scores in r2_scores_data[target].items():
                epochs = range(1, len(scores) + 1)
                ax.plot(epochs, scores, 
                    label=model_name.upper(), 
                    color=model_colors[model_name],
                    marker='o',
                    markersize=4,
                    linewidth=2,
                    alpha=0.7)
            
            # 设置图表样式
            ax.set_title(f'{target} Medal R² Score Trends')
            ax.set_xlabel('Validation Fold')
            ax.set_ylabel('R² Score')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 添加水平基准线
            ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
            ax.text(0, 0.81, 'R²=0.8 baseline', fontsize=8, alpha=0.7)
            
            # 设置y轴范围
            ax.set_ylim(0.5, 1.0)
        
        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'r2_scores_trends')
        
        return fig