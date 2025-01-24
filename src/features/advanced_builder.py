# src/features/advanced_builder.py
import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from rich.progress import Progress, SpinnerColumn
from rich.panel import Panel
from rich.console import Console
from rich.tree import Tree
from scipy import stats

class AdvancedFeatureBuilder:
    def __init__(self, medals_df, athletes_df, hosts_df, programs_df, console=None):
        self.medals_df = medals_df
        self.athletes_df = athletes_df
        self.hosts_df = hosts_df
        self.programs_df = programs_df
        self.console = console or Console()
        self.feature_stats = {}
        
    def _calculate_rolling_statistics(self, ts_data, windows=[3, 4, 8]):
        """计算滚动统计特征"""
        rolling_stats = pd.DataFrame()
        
        for window in windows:
            # 计算滚动平均
            rolling_mean = ts_data.rolling(window=window).mean()
            rolling_stats[f'rolling_mean_{window}'] = rolling_mean.mean(axis=1)
            
            # 计算滚动标准差
            rolling_std = ts_data.rolling(window=window).std()
            rolling_stats[f'rolling_std_{window}'] = rolling_std.mean(axis=1)
            
            # 计算滚动趋势
            rolling_trend = ts_data.rolling(window=window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0]
            )
            rolling_stats[f'rolling_trend_{window}'] = rolling_trend.mean(axis=1)
        
        return rolling_stats
    
    def _calculate_historical_performance(self):
        """计算历史表现指标"""
        historical = pd.DataFrame()
        
        try:
            # 确保使用年份作为索引
            medal_data = self.medals_df.copy()
            
            if 'Year' not in medal_data.columns:
                self.console.print("[red]错误：medals_df中没有Year列[/red]")
                return pd.DataFrame()
                
            # 验证数据
            if medal_data.empty:
                self.console.print("[red]错误：medals_df为空[/red]")
                return pd.DataFrame()
                
            # 按年份和国家分组计算
            grouped = medal_data.groupby(['Year', 'NOC']).agg({
                'Gold': 'sum',
                'Total': 'sum'
            }).reset_index()
            
            # 按年份排序
            grouped = grouped.sort_values('Year')
            
            # 为每个国家计算累计和增长指标
            historical_data = []
            
            for noc in grouped['NOC'].unique():
                country_data = grouped[grouped['NOC'] == noc].copy()
                if len(country_data) < 1:
                    continue
                    
                # 安全计算累计值
                country_data['cumulative_gold'] = country_data['Gold'].cumsum()
                country_data['cumulative_total'] = country_data['Total'].cumsum()
                
                # 计算增长率
                if len(country_data) > 1:
                    first_gold = country_data['Gold'].iloc[0]
                    first_total = country_data['Total'].iloc[0]
                    last_gold = country_data['Gold'].iloc[-1]
                    last_total = country_data['Total'].iloc[-1]
                    
                    growth_gold = (last_gold - first_gold) / first_gold if first_gold != 0 else 0
                    growth_total = (last_total - first_total) / first_total if first_total != 0 else 0
                else:
                    growth_gold = 0
                    growth_total = 0
                
                # 计算稳定性指标
                stability_gold = country_data['Gold'].std() / (country_data['Gold'].mean() if country_data['Gold'].mean() != 0 else 1)
                stability_total = country_data['Total'].std() / (country_data['Total'].mean() if country_data['Total'].mean() != 0 else 1)
                
                # 添加计算结果
                for _, row in country_data.iterrows():
                    historical_data.append({
                        'Year': row['Year'],
                        'NOC': noc,
                        'cumulative_gold': row['cumulative_gold'],
                        'cumulative_total': row['cumulative_total'],
                        'growth_rate_gold': growth_gold,
                        'growth_rate_total': growth_total,
                        'stability_gold': stability_gold,
                        'stability_total': stability_total
                    })
            
            # 创建DataFrame
            if historical_data:
                historical = pd.DataFrame(historical_data)
                historical = historical.set_index('Year')
                
                # 验证输出
                self.console.print(f"[green]历史表现指标计算完成，形状: {historical.shape}[/green]")
            else:
                self.console.print("[yellow]警告：没有生成历史表现数据[/yellow]")
                
            return historical
            
        except Exception as e:
            self.console.print(f"[red]历史表现指标计算失败: {str(e)}[/red]")
            return pd.DataFrame()
    
    def _calculate_trajectory_features(self):
        """计算发展轨迹特征"""
        trajectory = pd.DataFrame()
        
        try:
            # 计算最近几届的趋势
            recent_years = self.medals_df[self.medals_df['Year'] >= 2000].copy()
            
            # 确保数据不为空且有足够的样本
            if len(recent_years) > 0:
                # 计算趋势斜率
                def calculate_trend(group):
                    if len(group) < 2:  # 检查数据点数量
                        return 0
                    x = np.arange(len(group))
                    y = group.values
                    try:
                        slope, _, _, _, _ = stats.linregress(x, y)
                        return slope if not np.isnan(slope) else 0
                    except Exception:
                        return 0
                
                trends = recent_years.pivot(index='Year', columns='NOC', values='Total')
                if not trends.empty and trends.shape[0] > 1:  # 确保有足够的时间点
                    trajectory['recent_trend'] = trends.apply(calculate_trend)
                    
                    # 计算加速度（二阶差分）
                    diffs = trends.diff().diff()
                    trajectory['acceleration'] = diffs.mean().fillna(0)
                else:
                    trajectory['recent_trend'] = 0
                    trajectory['acceleration'] = 0
            
            # 填充可能的空值
            trajectory = trajectory.fillna(0)
            
        except Exception as e:
            self.console.print(f"[red]轨迹特征计算错误: {str(e)}[/red]")
            trajectory['recent_trend'] = 0
            trajectory['acceleration'] = 0
        
        return trajectory
    
    def _calculate_diversity_metrics(self):
        """计算多样性指标"""
        diversity = pd.DataFrame()
        
        # 计算项目参与度
        participation = self.athletes_df.groupby('NOC')['Sport'].nunique()
        diversity['sport_diversity'] = participation
        
        # 计算项目成功率
        def calculate_success_rate(group):
            total_participation = len(group)
            medal_count = group['Medal'].notna().sum()
            return medal_count / total_participation if total_participation > 0 else 0
        
        success_rates = self.athletes_df.groupby(['NOC', 'Sport']).apply(calculate_success_rate)
        diversity['avg_success_rate'] = success_rates.groupby('NOC').mean()
        
        return diversity
    
    def _calculate_success_matrix(self):
        """计算项目成功率矩阵"""
        # 创建国家-项目矩阵
        success_matrix = pd.pivot_table(
            self.athletes_df,
            values='Medal',
            index='NOC',
            columns='Sport',
            aggfunc=lambda x: (x != 'No Medal').mean(),
            fill_value=0
        )
        
        return success_matrix
    
    def _analyze_sport_correlations(self):
        """分析项目相关性"""
        correlations = pd.DataFrame()
        
        # 计算项目之间的相关性
        sport_matrix = self._calculate_success_matrix()
        corr_matrix = sport_matrix.corr()
        
        # 提取主要相关性特征
        correlations['avg_correlation'] = corr_matrix.mean()
        correlations['max_correlation'] = corr_matrix.max()
        
        return correlations
    
    def _analyze_sport_trends(self):
        """分析项目发展趋势"""
        trends = pd.DataFrame()
        
        # 按年份统计项目发展
        yearly_sports = self.athletes_df.groupby(['Year', 'Sport']).size().unstack(fill_value=0)
        
        # 计算趋势
        def calculate_trend(series):
            x = np.arange(len(series))
            y = series.values
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        
        trends['sport_trend'] = yearly_sports.apply(calculate_trend)
        
        return trends
    
    def _create_country_sport_interactions(self):
        """创建国家-项目交互特征"""
        interactions = pd.DataFrame()
        
        # 计算每个国家在不同项目上的特长
        success_matrix = self._calculate_success_matrix()
        
        # 计算项目专长指数
        interactions['specialization_index'] = success_matrix.max(axis=1)
        interactions['diversity_index'] = success_matrix.apply(lambda x: (x > 0).sum(), axis=1)
        
        return interactions
    
    def _create_time_performance_interactions(self):
        """创建时间-表现交互特征"""
        interactions = pd.DataFrame()
        
        # 计算时间趋势
        yearly_performance = self.medals_df.pivot(index='Year', columns='NOC', values='Total')
        
        # 计算近期表现权重
        recent_weight = np.linspace(0.5, 1, len(yearly_performance))
        weighted_performance = yearly_performance.multiply(recent_weight, axis=0)
        
        interactions['weighted_trend'] = weighted_performance.mean()
        
        return interactions
    
    def _create_host_effect_interactions(self):
        """创建主办国效应交互特征"""
        interactions = pd.DataFrame()
        
        # 合并主办国信息
        merged = pd.merge(self.medals_df, self.hosts_df, on='Year', how='left')
        
        # 计算主办国效应
        def calculate_host_effect(group):
            host_years = group['NOC'] == group['Host']
            if host_years.any():
                host_performance = group.loc[host_years, 'Total'].mean()
                normal_performance = group.loc[~host_years, 'Total'].mean()
                return host_performance / normal_performance if normal_performance > 0 else 0
            return 0
        
        interactions['host_effect'] = merged.groupby('NOC').apply(calculate_host_effect)
        
        return interactions
    
    def _calculate_statistical_moments(self):
        """计算统计矩"""
        moments = pd.DataFrame()
        
        try:
            # 计算高阶统计量
            for col in ['Gold', 'Total']:
                data = self.medals_df.pivot(index='Year', columns='NOC', values=col)
                
                # 安全计算偏度和峰度
                skew = data.skew().replace([np.inf, -np.inf], 0).fillna(0)
                kurt = data.kurtosis().replace([np.inf, -np.inf], 0).fillna(0)
                
                moments[f'{col}_skewness'] = skew
                moments[f'{col}_kurtosis'] = kurt
            
            # 确保没有无穷大值
            moments = moments.replace([np.inf, -np.inf], 0)
            moments = moments.fillna(0)
            
        except Exception as e:
            self.console.print(f"[red]统计矩计算错误: {str(e)}[/red]")
            # 创建空的统计量
            for col in ['Gold', 'Total']:
                moments[f'{col}_skewness'] = 0
                moments[f'{col}_kurtosis'] = 0
        
        return moments
    
    def _calculate_distribution_features(self):
        """计算分布特征"""
        distribution = pd.DataFrame()
        
        for col in ['Gold', 'Total']:
            data = self.medals_df.pivot(index='Year', columns='NOC', values=col)
            
            # 计算分位数特征
            distribution[f'{col}_q25'] = data.quantile(0.25)
            distribution[f'{col}_q75'] = data.quantile(0.75)
            distribution[f'{col}_iqr'] = distribution[f'{col}_q75'] - distribution[f'{col}_q25']
        
        return distribution
    
    def _calculate_volatility_metrics(self):
        """计算波动性指标"""
        volatility = pd.DataFrame()
        
        for col in ['Gold', 'Total']:
            data = self.medals_df.pivot(index='Year', columns='NOC', values=col)
            
            # 计算波动率
            volatility[f'{col}_volatility'] = data.std() / data.mean()
            
            # 计算最大回撤
            def calculate_max_drawdown(series):
                roll_max = series.expanding().max()
                drawdown = (series - roll_max) / roll_max
                return drawdown.min()
            
            volatility[f'{col}_max_drawdown'] = data.apply(calculate_max_drawdown)
        
        return volatility
    
    def _calculate_feature_importance(self, feature_df):
        """安全计算特征重要性"""
        try:
            if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
                numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    X = feature_df[numeric_cols].fillna(0)
                    # 清理数据，替换无穷大值
                    X = X.replace([np.inf, -np.inf], 0)
                    
                    y = self.medals_df.groupby('NOC')['Total'].mean()
                    
                    # 确保X和y的索引匹配
                    common_index = X.index.intersection(y.index)
                    if len(common_index) > 1:  # 确保有足够的样本
                        X = X.loc[common_index]
                        y = y.loc[common_index]
                        
                        # 使用稳健的特征选择方法
                        selector = SelectKBest(score_func=mutual_info_regression, k='all')
                        selector.fit(X, y)
                        
                        importance = pd.Series(selector.scores_, index=numeric_cols)
                        return importance.sort_values(ascending=False)
            
            return pd.Series()
            
        except Exception as e:
            self.console.print(f"[yellow]特征重要性计算失败: {str(e)}[/yellow]")
            return pd.Series()
    def build_all_features(self):
        """构建完整特征集"""
        self.console.print(Panel("[bold cyan]开始高级特征工程[/bold cyan]"))
        features = {}
        
        try:
            with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:
                task = progress.add_task("构建特征...", total=5)
                
                # 1. 时间序列特征
                self.console.print("[yellow]构建时间序列特征...[/yellow]")
                try:
                    ts_features = self._build_advanced_time_series_features()
                    features['time_series'] = ts_features
                    self.console.print("[green]✓ 时间序列特征构建完成[/green]")
                except Exception as e:
                    self.console.print(f"[red]时间序列特征构建失败: {str(e)}[/red]")
                progress.update(task, advance=1)
                
                # 2. 国家特征
                self.console.print("[yellow]构建国家特征...[/yellow]")
                try:
                    # 首先构建基础历史表现特征
                    historical_features = self._calculate_historical_performance()
                    if not historical_features.empty:
                        self.console.print("[green]✓ 历史表现特征构建完成[/green]")
                        
                        # 构建其他特征并确保索引一致性
                        trajectory_features = self._calculate_trajectory_features()
                        diversity_features = self._calculate_diversity_metrics()
                        
                        # 重新索引其他特征以匹配历史特征的年份
                        if not trajectory_features.empty:
                            trajectory_features = trajectory_features.reindex(historical_features.index)
                        if not diversity_features.empty:
                            diversity_features = diversity_features.reindex(historical_features.index)
                        
                        # 合并所有特征
                        country_features = historical_features
                        if not trajectory_features.empty:
                            country_features = country_features.join(trajectory_features, how='left')
                        if not diversity_features.empty:
                            country_features = country_features.join(diversity_features, how='left')
                        
                        features['country'] = country_features
                        self.console.print(f"[green]✓ 国家特征构建完成，形状: {country_features.shape}[/green]")
                        self.console.print(f"[cyan]特征索引范围: {country_features.index.min()} - {country_features.index.max()}[/cyan]")
                except Exception as e:
                    self.console.print(f"[red]国家特征构建失败: {str(e)}[/red]")
                    self.console.print(f"[red]错误详情: {str(e.__traceback__)}[/red]")
                progress.update(task, advance=1)
                
                # 3. 运动项目特征
                self.console.print("[yellow]构建运动项目特征...[/yellow]")
                try:
                    sport_features = pd.concat([
                        self._calculate_success_matrix(),
                        self._analyze_sport_correlations(),
                        self._analyze_sport_trends()
                    ], axis=1)
                    features['sport'] = sport_features
                    self.console.print("[green]✓ 运动项目特征构建完成[/green]")
                except Exception as e:
                    self.console.print(f"[red]运动项目特征构建失败: {str(e)}[/red]")
                progress.update(task, advance=1)
                
                # 4. 交互特征
                self.console.print("[yellow]构建交互特征...[/yellow]")
                try:
                    interaction_features = pd.concat([
                        self._create_country_sport_interactions(),
                        self._create_time_performance_interactions(),
                        self._create_host_effect_interactions()
                    ], axis=1)
                    features['interaction'] = interaction_features
                    self.console.print("[green]✓ 交互特征构建完成[/green]")
                except Exception as e:
                    self.console.print(f"[red]交互特征构建失败: {str(e)}[/red]")
                progress.update(task, advance=1)
                
                # 5. 统计特征
                self.console.print("[yellow]构建统计特征...[/yellow]")
                try:
                    statistical_features = pd.concat([
                        self._calculate_statistical_moments(),
                        self._calculate_distribution_features(),
                        self._calculate_volatility_metrics()
                    ], axis=1)
                    features['statistical'] = statistical_features
                    self.console.print("[green]✓ 统计特征构建完成[/green]")
                except Exception as e:
                    self.console.print(f"[red]统计特征构建失败: {str(e)}[/red]")
                progress.update(task, advance=1)
                
        except Exception as e:
            self.console.print(f"[bold red]特征工程过程中出错: {str(e)}[/bold red]")
            raise
        
        # 评估特征质量
        try:
            self._evaluate_features(features)
            self._display_feature_summary()
        except Exception as e:
            self.console.print(f"[red]特征评估过程中出错: {str(e)}[/red]")
        
        # 特征构建完成统计
        total_features = sum(f.shape[1] for f in features.values() if isinstance(f, pd.DataFrame))
        self.console.print(Panel(
            f"[bold green]特征工程完成[/bold green]\n"
            f"总特征数: {total_features}\n"
            f"特征类别数: {len(features)}",
            subtitle="✨ 特征构建成功"
        ))
        
        return features

    def _build_advanced_time_series_features(self):
        """构建高级时间序列特征"""
        # 准备时间序列数据
        ts_data = self.medals_df.pivot(index='Year', columns='NOC', values='Total').fillna(0)
        
        # 设置tsfresh参数
        fc_parameters = MinimalFCParameters()
        
        # 提取时间序列特征
        ts_features = extract_features(
            ts_data.reset_index(),
            column_id='Year',
            column_sort='Year',
            default_fc_parameters=fc_parameters,
            n_jobs=4  # 并行处理
        )
        
        # 添加滚动统计特征
        rolling_stats = self._calculate_rolling_statistics(ts_data)
        
        return pd.concat([ts_features, rolling_stats], axis=1)
    def _evaluate_features(self, features):
        """评估特征质量"""
        for feature_type, feature_df in features.items():
            try:
                if not isinstance(feature_df, pd.DataFrame):
                    self.console.print(f"[yellow]警告: {feature_type} 不是DataFrame格式，跳过评估[/yellow]")
                    continue
                
                if feature_df.empty:
                    self.console.print(f"[yellow]警告: {feature_type} 是空DataFrame，跳过评估[/yellow]")
                    continue
                    
                stats = {
                    'feature_count': feature_df.shape[1],
                    'memory_usage': feature_df.memory_usage(deep=True).sum() / 1024**2,  # MB
                    'missing_ratio': feature_df.isnull().sum().mean() / len(feature_df) if len(feature_df) > 0 else 0,
                }
                
                # 数值特征的统计信息
                numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # 安全计算统计量，处理无穷大值
                    df_clean = feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                    
                    stats.update({
                        'mean_std': df_clean.std().mean(),
                        'mean_skew': df_clean.skew().mean(),
                        'mean_kurtosis': df_clean.kurtosis().mean(),
                    })
                    
                    # 特征重要性计算
                    if hasattr(self, 'medals_df') and len(df_clean) > 3:  # 确保有足够的样本
                        try:
                            importance = self._calculate_feature_importance(df_clean)
                            if not importance.empty:
                                stats['top_features'] = importance.nlargest(min(5, len(importance))).index.tolist()
                                stats['importance_scores'] = importance.nlargest(min(5, len(importance))).values.tolist()
                        except Exception as e:
                            self.console.print(f"[yellow]特征重要性计算失败: {str(e)}[/yellow]")
                
                # 类别特征的统计信息
                categorical_cols = feature_df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    stats['categorical_features'] = len(categorical_cols)
                    stats['mean_unique_values'] = feature_df[categorical_cols].nunique().mean()
                
                # 相关性分析（对清理后的数据进行）
                if len(numeric_cols) > 1:
                    df_clean = feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                    corr_matrix = df_clean.corr().fillna(0)
                    upper_tri = np.triu(corr_matrix, k=1)
                    stats['mean_correlation'] = np.abs(upper_tri[upper_tri != 0]).mean()
                    high_corr_pairs = np.sum(np.abs(upper_tri) > 0.9)
                    stats['high_correlation_pairs'] = high_corr_pairs
                
                self.feature_stats[feature_type] = stats
                
            except Exception as e:
                self.console.print(f"[red]评估 {feature_type} 特征时出错: {str(e)}[/red]")
                # 添加基本统计信息
                self.feature_stats[feature_type] = {
                    'feature_count': feature_df.shape[1] if isinstance(feature_df, pd.DataFrame) else 0,
                    'memory_usage': 0,
                    'missing_ratio': 0
                }

    def _display_feature_summary(self):
        """显示特征评估摘要"""
        summary_tree = Tree("\n[bold cyan]特征工程评估报告[/bold cyan]")
        
        for feature_type, stats in self.feature_stats.items():
            # 创建特征类型分支
            feature_branch = summary_tree.add(f"[yellow]{feature_type}[/yellow]")
            
            # 基本信息
            feature_branch.add(f"特征数量: {stats['feature_count']}")
            feature_branch.add(f"内存占用: {stats['memory_usage']:.2f} MB")
            feature_branch.add(f"缺失值比例: {stats['missing_ratio']:.2%}")
            
            # 数值特征统计
            if 'mean_std' in stats:
                stats_branch = feature_branch.add("统计信息")
                stats_branch.add(f"平均标准差: {stats['mean_std']:.3f}")
                stats_branch.add(f"平均偏度: {stats['mean_skew']:.3f}")
                stats_branch.add(f"平均峰度: {stats['mean_kurtosis']:.3f}")
            
            # 相关性信息
            if 'mean_correlation' in stats:
                corr_branch = feature_branch.add("相关性分析")
                corr_branch.add(f"平均相关系数: {stats['mean_correlation']:.3f}")
                corr_branch.add(f"高相关特征对数量: {stats['high_correlation_pairs']}")
            
            # 重要特征信息
            if 'top_features' in stats:
                importance_branch = feature_branch.add("重要特征")
                for feat, score in zip(stats['top_features'], stats['importance_scores']):
                    importance_branch.add(f"{feat}: {score:.3f}")
            
            # 类别特征信息
            if 'categorical_features' in stats:
                cat_branch = feature_branch.add("类别特征")
                cat_branch.add(f"类别特征数: {stats['categorical_features']}")
                cat_branch.add(f"平均唯一值数: {stats['mean_unique_values']:.1f}")
        
        # 添加总体评估
        overall_branch = summary_tree.add("[bold green]总体评估[/bold green]")
        total_features = sum(stats['feature_count'] for stats in self.feature_stats.values())
        total_memory = sum(stats['memory_usage'] for stats in self.feature_stats.values())
        overall_branch.add(f"总特征数: {total_features}")
        overall_branch.add(f"总内存占用: {total_memory:.2f} MB")
        
        # 显示完整报告
        self.console.print(summary_tree)