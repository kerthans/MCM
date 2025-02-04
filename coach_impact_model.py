import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
from src.visualization.CoachAnalysisVisualizer import CoachAnalysisVisualizer 
@dataclass
class CoachImpact:
    sport: str
    country: str
    period: Tuple[int, int]
    medal_change: float
    significance: float
    consistency: float

class CoachImpactAnalyzer:
    def __init__(self):
        self.console = Console()
        self.coach_effects = {}
        self.country_recommendations = {}
        self.visualizer = CoachAnalysisVisualizer()

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载和准备分析所需的数据"""
        try:
            def try_load_data(file_path_base):
                """尝试多种方式加载数据"""
                # 确保基础路径是Path对象
                base_path = Path(file_path_base)
                data_dir = base_path.parent
                
                # 如果目录不存在，尝试创建
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # 定义加载尝试列表
                attempts = [
                    (base_path.with_suffix('.parquet'), lambda x: pd.read_parquet(x)),
                    (base_path.with_suffix('.csv'), lambda x: pd.read_csv(x)),
                    (base_path.with_suffix('.csv'), lambda x: pd.read_csv(x, encoding='utf-8')),
                    (base_path.with_suffix('.csv'), lambda x: pd.read_csv(x, encoding='latin1')),
                    (base_path.parent / f"{base_path.name}.parquet", lambda x: pd.read_parquet(x)),
                    (base_path.parent / f"{base_path.name}.csv", lambda x: pd.read_csv(x))
                ]
                
                errors = []
                for file_path, reader in attempts:
                    try:
                        if file_path.exists():
                            data = reader(file_path)
                            self.console.print(f"[green]成功从 {file_path} 加载数据[/green]")
                            return data
                    except Exception as e:
                        errors.append(f"{file_path}: {str(e)}")
                        continue
                
                # 检查目录内容
                available_files = list(data_dir.glob("*")) if data_dir.exists() else []
                files_str = "\n".join(f"- {f.name}" for f in available_files) if available_files else "目录为空"
                
                error_msg = (
                    f"无法加载数据文件 {base_path}.*\n"
                    f"尝试的路径:\n{chr(10).join(f'- {err}' for err in errors)}\n"
                    f"目录 {data_dir} 中的文件:\n{files_str}"
                )
                raise FileNotFoundError(error_msg)
            
            # 加载数据
            athletes_df = try_load_data("data/processed/athletes")
            medals_df = try_load_data("data/processed/medal_counts")
            
            # 数据预处理
            athletes_df['Year'] = pd.to_numeric(athletes_df['Year'], errors='coerce')
            athletes_df = athletes_df.dropna(subset=['Year', 'NOC', 'Sport'])
            
            # 转换Medal列为数值
            medal_map = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
            athletes_df['Medal_Value'] = athletes_df['Medal'].map(medal_map).fillna(0)
            
            # 数据验证
            required_cols = {'athletes': ['Year', 'NOC', 'Sport', 'Medal'],
                            'medals': ['Year', 'NOC', 'Gold', 'Total']}
                            
            for df_name, df in [('athletes', athletes_df), ('medals', medals_df)]:
                missing_cols = [col for col in required_cols[df_name] if col not in df.columns]
                if missing_cols:
                    self.console.print(f"[yellow]警告: {df_name} 数据集的列: {', '.join(df.columns)}[/yellow]")
                    raise ValueError(f"{df_name} 数据集缺少必要的列: {', '.join(missing_cols)}")
            
            return athletes_df, medals_df
            
        except Exception as e:
            self.console.print(f"[bold red]数据加载错误: {str(e)}[/bold red]")
            raise

    def detect_coach_effect_periods(self, data: pd.DataFrame, country: str, sport: str) -> List[CoachImpact]:
        """优化后的教练效应检测,增强数值稳定性和错误处理"""
        impacts = []
        min_medals = 3
        min_gap = 12
        window_size = 12

        try:
            # 输入验证
            if data.empty or not isinstance(country, str) or not isinstance(sport, str):
                return impacts

            # 规范化国家代码
            country_map = {'URS': 'RUS', 'GDR': 'GER', 'FRG': 'GER', 'TCH': 'CZE'}
            normalized_country = country_map.get(country, country)

            # 数据预处理和验证
            country_data = data[
                (data['NOC'] == normalized_country) &
                (data['Sport'] == sport)
                ].copy()  # 创建副本避免警告

            if country_data.empty:
                return impacts

            # 年度表现聚合与验证
            yearly_perf = country_data.groupby(['Year', 'Team'], observed=True).agg({
                'Medal_Value': ['sum', 'count']
            }).groupby('Year').agg({
                ('Medal_Value', 'sum'): 'sum',
                ('Medal_Value', 'count'): 'sum'
            }).reset_index()

            # 数据验证
            if yearly_perf.empty or yearly_perf[('Medal_Value', 'sum')].max() < min_medals:
                return impacts

            # 异常值处理
            medal_values = yearly_perf[('Medal_Value', 'sum')].values
            q1, q3 = np.percentile(medal_values, [25, 75])
            iqr = q3 - q1
            outlier_threshold = q3 + 1.5 * iqr
            medal_values = np.clip(medal_values, 0, outlier_threshold)
            yearly_perf[('Medal_Value', 'sum')] = medal_values

            # 窗口分析
            for i in range(len(yearly_perf) - window_size + 1):
                try:
                    window = yearly_perf.iloc[i:i + window_size]
                    p1, p2 = window.iloc[:window_size // 2], window.iloc[window_size // 2:]

                    # 数值计算保护
                    medal_diff = float(p2[('Medal_Value', 'sum')].mean() - p1[('Medal_Value', 'sum')].mean())
                    participation_ratio = max(1e-10, p1[('Medal_Value', 'count')].mean())
                    participation_change = float(p2[('Medal_Value', 'count')].mean() / participation_ratio - 1)

                    period = (int(window['Year'].iloc[0]), int(window['Year'].iloc[-1]))

                    # 验证时期重叠
                    if abs(medal_diff) >= min_medals and not any(
                            abs(period[0] - p[0]) < min_gap or abs(period[1] - p[1]) < min_gap
                            for p in [(imp.period[0], imp.period[1]) for imp in impacts]
                    ):
                        # 统计显著性测试
                        p_value = self._bootstrap_significance_test(
                            p1[('Medal_Value', 'sum')].values,
                            p2[('Medal_Value', 'sum')].values,
                            n_bootstrap=3000
                        )

                        # 一致性评估
                        consistency = self._evaluate_performance_consistency(window)

                        if p_value < 0.05 and consistency > 0.4:
                            impacts.append(CoachImpact(
                                sport=sport,
                                country=normalized_country,
                                period=period,
                                medal_change=medal_diff,
                                significance=float(1 - p_value),
                                consistency=consistency
                            ))

                except Exception as e:
                    self._detect_numerical_issues(f"Window {i}", window[('Medal_Value', 'sum')].values)
                    continue

            return sorted(impacts, key=lambda x: abs(x.medal_change), reverse=True)

        except Exception as e:
            print(f"教练效应检测错误: {str(e)}")
            return impacts
    def _calculate_stability_score(self, period1: pd.DataFrame, period2: pd.DataFrame) -> float:
        """计算表现稳定性得分"""
        std1 = period1['sum'].std()
        std2 = period2['sum'].std()
        mean1 = period1['sum'].mean()
        mean2 = period2['sum'].mean()
        
        cv1 = std1 / (mean1 + 1)  # 变异系数
        cv2 = std2 / (mean2 + 1)
        
        stability_improvement = (1 / (1 + cv2)) - (1 / (1 + cv1))
        return np.clip(stability_improvement, -1, 1)
    def _bootstrap_significance_test(self, period1: np.ndarray, period2: np.ndarray, n_bootstrap: int = 3000) -> float:
        """使用Bootstrap方法进行显著性检验,增加数值稳定性"""
        if len(period1) < 2 or len(period2) < 2:
            return 1.0
            
        try:
            # 数据预处理
            period1 = np.clip(period1, 0, np.percentile(period1, 99))
            period2 = np.clip(period2, 0, np.percentile(period2, 99))
            
            observed_diff = np.mean(period2) - np.mean(period1)
            combined = np.concatenate([period1, period2])
            n1, n2 = len(period1), len(period2)
            
            # Bootstrap重采样
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                resampled = np.random.choice(combined, size=len(combined), replace=True)
                diff = np.mean(resampled[n1:]) - np.mean(resampled[:n1])
                bootstrap_diffs.append(diff)
            
            # 计算p值
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
            return float(np.clip(p_value, 1e-8, 1.0))
        except:
            return 1.0

    def _evaluate_performance_consistency(self, data: pd.DataFrame) -> float:
        """增强版表现一致性评估"""
        try:
            # 数据提取和验证
            medals = data[('Medal_Value', 'sum')].values.astype(np.float64)
            if len(medals) < 3:
                return 0.0

            # 异常值处理
            q1, q3 = np.percentile(medals, [25, 75])
            iqr = q3 - q1
            medals_cleaned = np.clip(medals, q1 - 1.5 * iqr, q3 + 1.5 * iqr)

            # 基础统计计算
            medals_mean = np.mean(medals_cleaned)
            medals_std = np.std(medals_cleaned)

            # 数值稳定性检查
            if medals_std < 1e-10 or np.isclose(medals_mean, 0, atol=1e-10):
                return 0.0

            # 标准化处理
            medals_scaled = (medals_cleaned - medals_mean) / (medals_std + 1e-10)
            years = np.arange(len(medals_scaled), dtype=np.float64)

            try:
                # 主要方法：SVD分解
                X = np.vstack([years, np.ones_like(years)]).T
                U, s, Vh = np.linalg.svd(X, full_matrices=False)

                # 条件数检查
                if np.max(s) / np.min(s) > 1e10:  # 病态矩阵检查
                    raise np.linalg.LinAlgError("Matrix is ill-conditioned")

                coef = Vh.T @ np.diag(1 / s) @ U.T @ medals_scaled
                trend = coef[0] * years + coef[1]

            except np.linalg.LinAlgError:
                # 备选方法：稳健回归
                slope, intercept = stats.theilslopes(medals_scaled, years)[:2]
                trend = slope * years + intercept

            # 残差分析
            residuals = medals_scaled - trend
            residuals_std = np.std(residuals)

            # 一致性得分计算
            base_consistency = 1.0 / (1.0 + residuals_std)

            # 趋势权重
            trend_weight = np.abs(np.corrcoef(years, medals_scaled)[0, 1])

            # 最终一致性得分
            final_consistency = 0.7 * base_consistency + 0.3 * trend_weight

            return float(np.clip(final_consistency, 0, 1))

        except Exception as e:
            print(f"一致性评估错误: {str(e)}")
            return 0.0

    def _find_advanced_benchmark(self, data: pd.DataFrame, sport: str, target_country: str) -> Dict:
        """Improved benchmark country selection with dynamic window and decay"""

        def calculate_sport_strength(country_data: pd.DataFrame, window_years: int = 8) -> float:
            """Calculate sport strength with temporal weighting"""
            if len(country_data) < 3:
                return 0.0

            recent_data = country_data.sort_values('Year', ascending=False).head(window_years)
            if recent_data.empty:
                return 0.0

            # Exponential decay weights
            years = recent_data['Year'].values
            max_year = years.max()
            weights = np.exp(-0.2 * (max_year - years))

            # Weighted metrics
            weighted_medals = np.average(recent_data['Medal_Value'].values, weights=weights)
            weighted_trend = np.polyfit(range(len(recent_data)), recent_data['Medal_Value'].values, 1, w=weights)[0]

            return 0.7 * weighted_medals + 0.3 * weighted_trend

        def calculate_historical_fit(country_data: pd.DataFrame) -> float:
            """Calculate historical fit score with peak performance consideration"""
            if len(country_data) < 3:
                return 0.0

            years = country_data['Year'].values
            medals = country_data['Medal_Value'].values

            # Historical peak
            peak_medals = np.percentile(medals, 95)

            # Recent performance (last 3 cycles)
            recent_mask = years >= (years.max() - 12)
            recent_medals = medals[recent_mask] if any(recent_mask) else medals

            if len(recent_medals) == 0:
                return 0.0

            recent_avg = np.mean(recent_medals)

            # Consistency factor
            consistency = 1 / (1 + np.std(recent_medals))

            # Combined score
            historical_fit = (0.4 * (recent_avg / peak_medals) +
                              0.4 * consistency +
                              0.2 * (len(country_data) / 20))  # Experience factor

            return float(np.clip(historical_fit, 0, 1))

        try:
            # Dynamic analysis period based on sport
            sport_cycles = {
                'Swimming': 8,
                'Athletics': 12,
                'Gymnastics': 16,
                'Volleyball': 8
            }
            analysis_period = sport_cycles.get(sport, 12)

            # Country code normalization
            country_mapping = {
                'URS': 'RUS', 'GDR': 'GER', 'FRG': 'GER',
                'TCH': 'CZE', 'YUG': 'SRB'
            }
            normalized_country = country_mapping.get(target_country, target_country)

            # Get target country data
            target_data = data[
                (data['Sport'] == sport) &
                (data['Year'] >= data['Year'].max() - analysis_period) &
                (data['NOC'] == normalized_country)
                ]

            if target_data.empty:
                return None

            # Calculate target metrics
            target_strength = calculate_sport_strength(target_data)
            target_fit = calculate_historical_fit(target_data)

            # Calculate current performance with decay
            recent_performance = target_data.sort_values('Year', ascending=False)['Medal_Value'].head(3).mean()
            if recent_performance == 0:
                # Find last medal year and apply decay
                last_medal = data[
                    (data['Sport'] == sport) &
                    (data['NOC'] == normalized_country) &
                    (data['Medal_Value'] > 0)
                    ].sort_values('Year', ascending=False)

                if not last_medal.empty:
                    years_since = data['Year'].max() - last_medal['Year'].iloc[0]
                    decay_factor = 0.8 ** (years_since / 4)  # 20% decay per cycle
                    recent_performance = last_medal['Medal_Value'].iloc[0] * decay_factor

            # Find benchmark countries
            other_countries = []
            for country in data[data['NOC'] != normalized_country]['NOC'].unique():
                if country in country_mapping.keys():  # Skip historical countries
                    continue

                country_data = data[
                    (data['Sport'] == sport) &
                    (data['Year'] >= data['Year'].max() - analysis_period) &
                    (data['NOC'] == country)
                    ]

                if len(country_data) >= 3:
                    strength = calculate_sport_strength(country_data)
                    hist_fit = calculate_historical_fit(country_data)

                    if strength > target_strength:
                        performance = country_data.sort_values('Year', ascending=False)['Medal_Value'].head(3).mean()

                        # Monte Carlo simulation for uncertainty
                        volatility = {'Swimming': 0.15, 'Gymnastics': 0.25}.get(sport, 0.2)
                        simulations = np.random.normal(performance, performance * volatility, 1000)
                        ci = np.percentile(simulations, [5, 95])

                        other_countries.append({
                            'NOC': country,
                            'strength': strength,
                            'historical_fit': hist_fit,
                            'performance': performance,
                            'uncertainty': (ci[0], ci[1]),
                            'score': 0.4 * strength + 0.3 * hist_fit + 0.3 * (performance / (target_strength + 1e-6))
                        })

            if not other_countries:
                return None

            # Select best benchmark
            best_benchmark = max(other_countries, key=lambda x: x['score'])

            # Calculate normalized improvement potential
            sport_max = data.groupby('Sport')['Medal_Value'].max()[sport]
            improvement_potential = (best_benchmark['performance'] - recent_performance) / (sport_max + 1e-6)

            return {
                'sport': sport,
                'benchmark_country': str(best_benchmark['NOC']),
                'current_performance': float(recent_performance),
                'improvement_potential': float(np.clip(improvement_potential, 0, 1)),
                'historical_fit': float(target_fit),
                'estimated_medal_gain': {
                    'mean': float(best_benchmark['performance'] - recent_performance),
                    'range': f"{best_benchmark['uncertainty'][0]:.1f}-{best_benchmark['uncertainty'][1]:.1f}"
                },
                'benchmark_metrics': {
                    'avg_performance': float(best_benchmark['performance']),
                    'stability': float(best_benchmark['historical_fit']),
                    'experience_years': int(len(target_data))
                }
            }

        except Exception as e:
            print(f"Benchmark analysis error: {str(e)}")
            return None
    def _calculate_country_metrics(self, data: pd.DataFrame) -> Dict:
        """优化后的国家表现指标计算"""
        if data.empty:
            return {
                'avg_performance': 0.0,
                'recent_performance': 0.0,
                'stability': 0.0,
                'trend_score': 0.0,
                'consistency': 0.0,
                'experience': 0
            }
        
        try:
            years = data['Year'].values
            medals = data['Medal_Value'].values
            
            # 基础统计
            recent_perf = medals[-4:].mean() if len(medals) >= 4 else medals.mean()
            avg_perf = medals.mean()
            stability = 1 / (1 + np.std(medals))
            
            # 趋势分析，增加错误处理
            try:
                if len(years) > 2:
                    X = (years - years.min()).reshape(-1, 1)
                    y = medals.reshape(-1, 1)
                    slope = np.polyfit(X.ravel(), y.ravel(), 1, rcond=1e-10)[0]
                    trend_score = np.clip(slope / (avg_perf + 1), -1, 1)
                else:
                    trend_score = 0.0
            except:
                trend_score = 0.0
            
            # 一致性评估
            consistency = 1 - np.std(medals) / (avg_perf + 1)
            
            return {
                'avg_performance': avg_perf,
                'recent_performance': recent_perf,
                'stability': stability,
                'trend_score': float(trend_score),
                'consistency': consistency,
                'experience': len(years)
            }
        except Exception as e:
            return {
                'avg_performance': 0.0,
                'recent_performance': 0.0,
                'stability': 0.0,
                'trend_score': 0.0,
                'consistency': 0.0,
                'experience': 0
            }

    def analyze_great_coach_effect(self,
                                 athletes_df: pd.DataFrame,
                                 medals_df: pd.DataFrame) -> Dict:
        """分析"伟大教练"效应"""
        results = {}

        # 选择重点分析的运动项目
        focus_sports = ['Gymnastics', 'Swimming', 'Athletics', 'Volleyball']

        # 分析每个重点项目
        for sport in focus_sports:
            sport_results = {}

            # 获取该项目的主要参赛国
            top_countries = athletes_df[
                athletes_df['Sport'] == sport
            ]['NOC'].value_counts().head(10).index

            for country in top_countries:
                impacts = self.detect_coach_effect_periods(athletes_df, country, sport)
                if impacts:
                    sport_results[country] = impacts

            results[sport] = sport_results

        return results

    def recommend_coach_investments(self, athletes_df: pd.DataFrame, medals_df: pd.DataFrame,
                                    countries: List[str]) -> Dict:
        """增强版教练投资建议系统"""

        def analyze_historical_data(data: pd.DataFrame, country: str, sport: str) -> Dict:
            """分析历史数据并计算关键指标"""
            print(f"\n分析 {country} 在 {sport} 项目的历史数据:")

            # 检查原始数据
            print(f"数据集中unique的Sport值: {data['Sport'].unique()}")
            print(f"数据集中unique的NOC值: {data['NOC'].unique()}")

            # 放宽匹配条件，使用模糊匹配
            sport_mask = data['Sport'].str.contains(sport, case=False, na=False)
            country_mask = data['NOC'].str.contains(country, case=False, na=False)

            country_data = data[sport_mask & country_mask].copy()

            print(f"找到 {len(country_data)} 条原始数据记录")
            if len(country_data) == 0:
                print(f"尝试更广泛的搜索...")
                # 尝试查找可能的运动项目名称变体
                possible_sports = data[data['Sport'].str.contains(sport[:4], case=False, na=False)]['Sport'].unique()
                print(f"可能的运动项目: {possible_sports}")

                # 尝试查找可能的国家代码变体
                possible_countries = data[data['NOC'].str.contains(country[:2], case=False, na=False)]['NOC'].unique()
                print(f"可能的国家代码: {possible_countries}")

                # 使用更宽松的匹配
                sport_mask = data['Sport'].isin(possible_sports)
                country_mask = data['NOC'].isin(possible_countries)
                country_data = data[sport_mask & country_mask].copy()
                print(f"宽松匹配后找到 {len(country_data)} 条记录")

            if len(country_data) == 0:
                print(f"警告: {country} 在 {sport} 项目中没有数据")
                return create_empty_metrics()

            # 验证数据的完整性
            print("\n数据验证:")
            print(f"年份范围: {country_data['Year'].min()} - {country_data['Year'].max()}")
            print(f"Medal_Value统计: \n{country_data['Medal_Value'].describe()}")

            # 计算年度统计
            yearly_medals = country_data.groupby('Year')['Medal_Value'].sum()
            print(f"\n年度奖牌统计:\n{yearly_medals}")

            # 处理异常值
            q1, q3 = yearly_medals.quantile([0.25, 0.75])
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            yearly_medals = yearly_medals.clip(upper=upper_bound)

            # 计算近期表现 (最近8年)
            recent_years = yearly_medals.index >= yearly_medals.index.max() - 8
            recent_performance = yearly_medals[recent_years].mean() if any(recent_years) else 0
            print(f"近期平均表现: {recent_performance}")

            # 计算趋势
            if len(yearly_medals) >= 2:
                years = np.array(range(len(yearly_medals)))
                trend = np.polyfit(years, yearly_medals.values, 1)[0]
            else:
                trend = 0
            print(f"趋势系数: {trend}")

            # 计算历史适配度指标
            total_medals = yearly_medals.sum()
            n_years = len(yearly_medals)
            span_years = yearly_medals.index.max() - yearly_medals.index.min() + 4
            participation_rate = n_years * 4 / span_years if span_years > 0 else 0

            # 计算稳定性
            consistency = 1 / (1 + yearly_medals.std()) if len(yearly_medals) > 1 else 0

            print(f"""
        详细统计:
        - 总奖牌数: {total_medals}
        - 参赛年数: {n_years}
        - 参与率: {participation_rate:.2f}
        - 稳定性: {consistency:.2f}
        """)

            # 综合历史适配度
            historical_fit = (
                    0.4 * (total_medals / (n_years + 1)) / 10 +  # 归一化平均奖牌数
                    0.3 * participation_rate +
                    0.3 * consistency
            )

            metrics = {
                'peak': float(yearly_medals.max()),
                'recent': float(recent_performance),
                'trend': float(trend),
                'years': int(n_years),
                'historical_fit': float(historical_fit),
                'consistency': float(consistency),
                'total_medals': int(total_medals)
            }

            print(f"最终指标:\n{metrics}")
            return metrics

        def create_empty_metrics():
            """创建空指标"""
            return {
                'peak': 0,
                'recent': 0,
                'trend': 0,
                'years': 0,
                'historical_fit': 0,
                'consistency': 0,
                'total_medals': 0
            }

        def evaluate_improvement_potential(target: Dict, benchmark: Dict) -> float:
            """评估改进潜力"""
            print(f"\n评估改进潜力:")
            print(f"目标指标: {target}")
            print(f"标杆指标: {benchmark}")

            # 如果是新项目或历史表现较弱
            if target['total_medals'] < 3:
                base_potential = 0.3
                print(f"新项目或历史较弱，基础潜力: {base_potential}")
                return base_potential

            # 计算相对差距
            relative_gap = (benchmark['recent'] - target['recent']) / (benchmark['recent'] + 1)
            print(f"相对差距: {relative_gap}")

            # 趋势因子
            trend_factor = np.tanh(max(0, benchmark['trend'] - target['trend']))
            print(f"趋势因子: {trend_factor}")

            # 历史因子
            historical_factor = benchmark['historical_fit'] / (target['historical_fit'] + 0.1)
            print(f"历史因子: {historical_factor}")

            # 综合评分
            potential = (
                    0.5 * relative_gap +
                    0.3 * trend_factor +
                    0.2 * min(historical_factor, 2)  # 限制历史因子的影响
            )

            final_potential = float(np.clip(potential, 0, 1))
            print(f"最终潜力评分: {final_potential}")
            return final_potential

        try:
            recommendations = {}
            print(f"\n开始生成教练投资建议...")

            # 基础运动项目
            sports = ['Swimming', 'Athletics', 'Gymnastics', 'Volleyball']

            # 合理的最大奖牌预期
            max_medals = {
                'Swimming': 15,
                'Athletics': 12,
                'Gymnastics': 10,
                'Volleyball': 8
            }

            for country in countries:
                print(f"\n分析 {country} 的投资机会:")
                country_recommendations = []

                for sport in sports:
                    print(f"\n评估 {sport} 项目:")
                    try:
                        # 获取目标国家数据
                        target_metrics = analyze_historical_data(athletes_df, country, sport)

                        # 获取标杆国家数据
                        benchmark_data = athletes_df[
                            (athletes_df['Sport'] == sport) &
                            (athletes_df['NOC'] != country) &
                            (~athletes_df['NOC'].isin(['URS', 'GDR', 'FRG', 'TCH', 'YUG']))
                            ]

                        # 选择最佳表现的国家作为标杆
                        benchmark_country = benchmark_data.groupby('NOC')['Medal_Value'].sum().nlargest(1).index[0]
                        print(f"选择的标杆国家: {benchmark_country}")

                        benchmark_metrics = analyze_historical_data(athletes_df, benchmark_country, sport)

                        # 计算潜力
                        potential = evaluate_improvement_potential(target_metrics, benchmark_metrics)

                        # 计算预期奖牌增长
                        medal_gain = min(
                            max_medals[sport],
                            benchmark_metrics['recent'] - target_metrics['recent']
                        )

                        if potential >= 0.1:  # 降低潜力阈值
                            country_recommendations.append({
                                'sport': sport,
                                'benchmark_country': benchmark_country,
                                'current_performance': float(target_metrics['recent']),
                                'improvement_potential': float(potential),
                                'historical_fit': float(target_metrics['historical_fit']),
                                'estimated_medal_gain': {
                                    'mean': float(medal_gain),
                                    'range': f"{(medal_gain * 0.85):.1f}-{(medal_gain * 1.15):.1f}"
                                },
                                'benchmark_metrics': {
                                    'avg_performance': float(benchmark_metrics['recent']),
                                    'trend': float(benchmark_metrics['trend']),
                                    'experience_years': int(benchmark_metrics['years'])
                                }
                            })

                    except Exception as e:
                        print(f"处理 {sport} 时出错: {str(e)}")
                        continue

                recommendations[country] = sorted(
                    country_recommendations,
                    key=lambda x: x['improvement_potential'],
                    reverse=True
                )

            return recommendations

        except Exception as e:
            print(f"推荐系统错误: {str(e)}")
            return {}
    def _find_benchmark_country(self, 
                              data: pd.DataFrame, 
                              sport: str, 
                              exclude_country: str) -> str:
        """找到某个运动项目的标杆国家"""
        performance = data[
            (data['Sport'] == sport) & 
            (data['NOC'] != exclude_country)
        ].groupby('NOC')['Medal_Value'].mean()
        
        return performance.nlargest(1).index[0] if not performance.empty else None

    def _calculate_improvement_potential(self, 
                                      data: pd.DataFrame, 
                                      sport: str, 
                                      country: str, 
                                      benchmark_country: str) -> float:
        """计算潜在提升空间"""
        country_perf = data[
            (data['Sport'] == sport) & 
            (data['NOC'] == country)
        ]['Medal_Value'].mean()
        
        benchmark_perf = data[
            (data['Sport'] == sport) & 
            (data['NOC'] == benchmark_country)
        ]['Medal_Value'].mean()
        
        return max(0, benchmark_perf - country_perf)

    def generate_report(self, coach_effects: Dict, recommendations: Dict) -> str:
        """优化的报告生成器"""
        report_lines = []

        # 1. 教练效应分析
        report_lines.extend([
            "1. '伟大教练'效应分析",
            "-" * 50,
            ""
        ])

        for sport, countries_data in coach_effects.items():
            if countries_data:  # Only add sport section if there's data
                report_lines.append(f"\n{sport}:")
                for country, impacts in countries_data.items():
                    for impact in impacts:
                        report_lines.append(
                            f"  - {country} ({impact.period[0]}-{impact.period[1]}): "
                            f"奖牌变化 {impact.medal_change:.1f}, "
                            f"显著性 {impact.significance:.2f}, "
                            f"一致性 {impact.consistency:.2f}"
                        )

        # 2. 教练投资建议
        report_lines.extend([
            "\n\n2. 教练投资建议",
            "-" * 50,
            ""
        ])

        for country, recommendations_list in recommendations.items():
            report_lines.append(f"\n{country}的优先投资项目:")
            if recommendations_list:  # Check if there are recommendations
                for rec in recommendations_list:
                    report_lines.extend([
                        f"  - {rec['sport']}:",
                        f"    * 标杆国家: {rec['benchmark_country']}",
                        f"    * 当前水平: {rec['current_performance']:.2f}",
                        f"    * 提升潜力: {rec['improvement_potential']:.2f}",
                        f"    * 历史适配度: {rec['historical_fit']:.2f}",
                        f"    * 预期奖牌增长: {rec['estimated_medal_gain']['mean']:.2f} ({rec['estimated_medal_gain']['range']})",
                        f"    * 标杆国绩效:",
                        f"      - 平均表现: {rec['benchmark_metrics']['avg_performance']:.2f}",
                        f"      - 发展趋势: {rec['benchmark_metrics']['trend']:.2f}",
                        f"      - 发展年限: {rec['benchmark_metrics']['experience_years']}"
                    ])
            else:
                report_lines.append("  暂无优先投资建议")

        return "\n".join(report_lines)

    def _detect_numerical_issues(self, label: str, matrix: np.ndarray):
        """数值问题诊断"""
        issues = []
        if np.any(np.isnan(matrix)):
            issues.append(f"{label} 包含NaN")
        if np.any(np.isinf(matrix)):
            issues.append(f"{label} 包含Inf")
        if np.any(np.abs(matrix) < 1e-10):
            issues.append(f"{label} 包含接近零值")
        if len(issues) > 0:
            print(f"数值问题 - {', '.join(issues)}")
            print(f"矩阵统计: 形状{matrix.shape}, 最小值{np.min(matrix)}, 最大值{np.max(matrix)}")
def main():
    console = Console()
    
    try:
        # 初始化分析器
        analyzer = CoachImpactAnalyzer()
        
        # 加载数据
        console.print("[bold cyan]加载数据...[/bold cyan]")
        athletes_df, medals_df = analyzer.load_and_prepare_data()
        
        # 分析教练效应
        console.print("[bold cyan]分析'伟大教练'效应...[/bold cyan]")
        coach_effects = analyzer.analyze_great_coach_effect(athletes_df, medals_df)
        analyzer.visualizer.plot_coach_effect_analysis(coach_effects)
        # 为特定国家生成建议
        target_countries = ['France', 'Germany', 'Italy']  # 示例国家
        console.print("[bold cyan]生成教练投资建议...[/bold cyan]")
        recommendations = analyzer.recommend_coach_investments(
            athletes_df, medals_df, target_countries
        )
        analyzer.visualizer.plot_investment_recommendations(recommendations)
        # 生成报告
        report = analyzer.generate_report(coach_effects, recommendations)
        
        # 保存报告
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "coach_impact_analysis_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
        
        # 显示报告
        console.print("\n[bold green]分析报告:[/bold green]")
        console.print(report)
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()