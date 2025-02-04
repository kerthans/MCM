import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

class CoachAnalysisVisualizer:
    def __init__(self, output_dir="outputs/3"):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置颜色方案
        self.colors = {
            'positive': '#2ecc71',  # 绿色
            'negative': '#e74c3c',  # 红色
            'neutral': '#95a5a6',   # 灰色
            'highlight': '#3498db'  # 蓝色
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

    def plot_coach_effect_analysis(self, coach_effects: dict, save=True):
        """绘制教练效应分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # 1. 时间序列影响图
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_temporal_effects(ax1, coach_effects)

        # 2. 项目影响分布
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_sport_impact_distribution(ax2, coach_effects)

        # 3. 显著性分析
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_significance_analysis(ax3, coach_effects)

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'coach_effect_analysis')
        
        return fig

    def _plot_temporal_effects(self, ax, coach_effects):
        """绘制时间序列影响图"""
        data_points = []
        
        for sport, countries in coach_effects.items():
            for country, impacts in countries.items():
                for impact in impacts:
                    midpoint = sum(impact.period) / 2
                    data_points.append({
                        'Year': midpoint,
                        'Sport': sport,
                        'Country': country,
                        'Impact': impact.medal_change,
                        'Significance': impact.significance
                    })
        
        if not data_points:
            ax.text(0.5, 0.5, 'No temporal data available',
                   ha='center', va='center')
            return
            
        df = pd.DataFrame(data_points)
        
        # 绘制散点图
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] 
                 for x in df['Impact']]
        sizes = df['Significance'] * 100 + 50  # 根据显著性调整点的大小
        
        scatter = ax.scatter(df['Year'], df['Impact'], 
                           c=colors, s=sizes, alpha=0.6)
        
        # 添加趋势线
        z = np.polyfit(df['Year'], df['Impact'], 1)
        p = np.poly1d(z)
        ax.plot(df['Year'], p(df['Year']), "--", 
                color=self.colors['neutral'])
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Coach Impact Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Medal Change')

    def _plot_sport_impact_distribution(self, ax, coach_effects):
        """绘制项目影响分布图"""
        sport_impacts = {}
        
        for sport, countries in coach_effects.items():
            impacts = []
            for country, effect_list in countries.items():
                impacts.extend([e.medal_change for e in effect_list])
            if impacts:
                sport_impacts[sport] = np.mean(impacts)
        
        if not sport_impacts:
            ax.text(0.5, 0.5, 'No sport impact data available',
                   ha='center', va='center')
            return
            
        sports = list(sport_impacts.keys())
        impacts = list(sport_impacts.values())
        
        colors = [self.colors['positive'] if x > 0 else self.colors['negative'] 
                 for x in impacts]
        
        bars = ax.barh(sports, impacts, color=colors, alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}', ha='left' if width > 0 else 'right',
                   va='center')
        
        ax.set_title('Average Impact by Sport')
        ax.set_xlabel('Average Medal Change')

    def _plot_significance_analysis(self, ax, coach_effects):
        """绘制显著性分析图"""
        significance_data = []
        
        for sport, countries in coach_effects.items():
            for country, impacts in countries.items():
                for impact in impacts:
                    significance_data.append({
                        'Sport': sport,
                        'Significance': impact.significance,
                        'Impact': abs(impact.medal_change)
                    })
        
        if not significance_data:
            ax.text(0.5, 0.5, 'No significance data available',
                   ha='center', va='center')
            return
            
        df = pd.DataFrame(significance_data)
        
        sns.scatterplot(data=df, x='Significance', y='Impact',
                       hue='Sport', ax=ax, alpha=0.7)
        
        ax.set_title('Impact Significance Analysis')
        ax.set_xlabel('Statistical Significance')
        ax.set_ylabel('Absolute Impact')
        
        # 调整图例位置
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_investment_recommendations(self, recommendations: dict, save=True):
        """绘制投资建议分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # 1. 潜力矩阵
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_potential_matrix(ax1, recommendations)

        # 2. 预期收益分析
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_expected_returns(ax2, recommendations)

        # 3. 历史适配度分析
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_historical_fit(ax3, recommendations)

        plt.tight_layout()
        
        if save:
            self._save_figure(fig, 'investment_recommendations')
        
        return fig

    def _plot_potential_matrix(self, ax, recommendations):
        """绘制潜力矩阵"""
        matrix_data = []
        
        for country, recs in recommendations.items():
            for rec in recs:
                matrix_data.append({
                    'Country': country,
                    'Sport': rec['sport'],
                    'Potential': rec['improvement_potential']
                })
        
        if not matrix_data:
            ax.text(0.5, 0.5, 'No potential matrix data available',
                   ha='center', va='center')
            return
            
        df = pd.DataFrame(matrix_data)
        pivot_table = df.pivot(index='Country', columns='Sport', 
                             values='Potential')
        
        sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax,
                   annot=True, fmt='.2f', 
                   cbar_kws={'label': 'Improvement Potential'})
        
        ax.set_title('Investment Potential Matrix')

    def _plot_expected_returns(self, ax, recommendations):
        """绘制预期收益分析图"""
        return_data = []
        
        for country, recs in recommendations.items():
            for rec in recs:
                mean_gain = float(rec['estimated_medal_gain']['mean'])
                return_data.append({
                    'Country': country,
                    'Sport': rec['sport'],
                    'Expected_Return': mean_gain
                })
        
        if not return_data:
            ax.text(0.5, 0.5, 'No return data available',
                   ha='center', va='center')
            return
            
        df = pd.DataFrame(return_data)
        
        sns.barplot(data=df, x='Country', y='Expected_Return', 
                   hue='Sport', ax=ax)
        
        ax.set_title('Expected Medal Returns')
        ax.set_xlabel('Country')
        ax.set_ylabel('Expected Medal Gain')
        ax.tick_params(axis='x', rotation=45)
        
        # 调整图例位置
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_historical_fit(self, ax, recommendations):
        """绘制历史适配度分析图"""
        fit_data = []
        
        for country, recs in recommendations.items():
            for rec in recs:
                fit_data.append({
                    'Country': country,
                    'Sport': rec['sport'],
                    'Historical_Fit': rec['historical_fit']
                })
        
        if not fit_data:
            ax.text(0.5, 0.5, 'No historical fit data available',
                   ha='center', va='center')
            return
            
        df = pd.DataFrame(fit_data)
        
        sns.boxplot(data=df, x='Country', y='Historical_Fit', 
                   ax=ax, color=self.colors['highlight'])
        
        ax.set_title('Historical Fit Analysis')
        ax.set_xlabel('Country')
        ax.set_ylabel('Historical Fit Score')
        ax.tick_params(axis='x', rotation=45)

    def _save_figure(self, fig, name):
        """保存图表"""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return path