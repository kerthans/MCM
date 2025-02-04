import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from src.visualization.OlympicInsightsVisualizer import OlympicInsightsVisualizer
@dataclass
class CountryInsight:
    noc: str
    trend_score: float
    stability_score: float
    diversity_score: float
    key_findings: List[str]
    recommendations: List[str]

class OlympicMedalInsightAnalyzer:
    def __init__(self):
        self.console = Console()
        self.insights = {}
        self.trends = {}
        self.patterns = {}
        self.visualizer = OlympicInsightsVisualizer()
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载和准备分析所需的数据"""
        try:
            def try_load_data(file_path_base: str) -> pd.DataFrame:
                """尝试多种方式加载数据"""
                base_path = Path(file_path_base)
                data_dir = base_path.parent
                data_dir.mkdir(parents=True, exist_ok=True)
                
                attempts = [
                    (base_path.with_suffix('.parquet'), lambda x: pd.read_parquet(x)),
                    (base_path.with_suffix('.csv'), lambda x: pd.read_csv(x)),
                    (base_path.with_suffix('.csv'), lambda x: pd.read_csv(x, encoding='utf-8')),
                    (base_path.with_suffix('.csv'), lambda x: pd.read_csv(x, encoding='latin1')),
                    (base_path.parent / f"{base_path.name}.parquet", lambda x: pd.read_parquet(x)),
                    (base_path.parent / f"{base_path.name}.csv", lambda x: pd.read_csv(x)),
                    (base_path.parent / f"{base_path.name}.xlsx", lambda x: pd.read_excel(x))
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
                
                available_files = list(data_dir.glob("*")) if data_dir.exists() else []
                files_str = "\n".join(f"- {f.name}" for f in available_files) if available_files else "目录为空"
                
                error_msg = (
                    f"无法加载数据文件 {base_path}.*\n"
                    f"尝试的路径:\n{chr(10).join(f'- {err}' for err in errors)}\n"
                    f"目录 {data_dir} 中的文件:\n{files_str}"
                )
                raise FileNotFoundError(error_msg)

            # 加载数据
            medals_df = try_load_data("data/processed/medal_counts")
            athletes_df = try_load_data("data/processed/athletes")
            programs_df = try_load_data("data/processed/programs")
            
            # 数据预处理
            for df in [medals_df, athletes_df, programs_df]:
                if 'Year' in df.columns:
                    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                    
            # 移除重复项
            medals_df = medals_df.drop_duplicates()
            athletes_df = athletes_df.drop_duplicates()
            programs_df = programs_df.drop_duplicates()
            
            # 验证必要的列
            required_cols = {
                'medals_df': ['Year', 'NOC', 'Gold', 'Total'],
                'athletes_df': ['Year', 'NOC', 'Sport'],
                'programs_df': ['Sport', 'Discipline']
            }
            
            for df_name, df in [
                ('medals_df', medals_df), 
                ('athletes_df', athletes_df), 
                ('programs_df', programs_df)
            ]:
                missing_cols = [col for col in required_cols[df_name] if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"{df_name} 缺少必要的列: {', '.join(missing_cols)}")
            
            return medals_df, athletes_df, programs_df
            
        except Exception as e:
            self.console.print(f"[bold red]数据加载错误: {str(e)}[/bold red]")
            raise

    def analyze_medal_trends(self, medals_df: pd.DataFrame, athletes_df: pd.DataFrame) -> Dict:
        """分析奖牌趋势和模式"""
        # 预处理：过滤历史国家
        historical_nocs = {
            'URS', 'GDR', 'FRG', 'EUN', 'YUG', 'TCH', 'ROC',
            'Soviet Union', 'East Germany', 'West Germany', 'Unified Team'
        }
        medals_df = medals_df[~medals_df['NOC'].isin(historical_nocs)].copy()
        athletes_df = athletes_df[~athletes_df['NOC'].isin(historical_nocs)].copy()
        trends = {}
        
        # 1. 整体趋势分析
        overall_trends = {
            'total_countries': medals_df['NOC'].nunique(),
            'medals_concentration': self._calculate_medals_concentration(medals_df),
            'emerging_countries': self._identify_emerging_countries(medals_df),
            'declining_countries': self._identify_declining_countries(medals_df)
        }
        trends['overall'] = overall_trends
        
        # 2. 区域性分析
        region_mapping = self._create_region_mapping()
        medals_df['Region'] = medals_df['NOC'].map(region_mapping)
        regional_trends = self._analyze_regional_patterns(medals_df)
        trends['regional'] = regional_trends
        
        # 3. 项目多样性分析 - 使用 athletes_df 而不是 medals_df
        diversity_trends = self._analyze_sport_diversity(athletes_df)
        trends['diversity'] = diversity_trends
        
        return trends

    def _calculate_medals_concentration(self, data: pd.DataFrame) -> Dict:
        """计算奖牌集中度趋势"""
        concentration = {}
        
        # 按时期分析
        periods = [(1896, 1950), (1951, 2000), (2001, 2024)]
        
        for start, end in periods:
            period_data = data[(data['Year'] >= start) & (data['Year'] <= end)]
            total_medals = period_data.groupby('NOC')['Total'].sum()
            
            # 计算基尼系数
            gini = self._calculate_gini(total_medals.values)
            
            # 计算前10国家占比
            top_10_share = total_medals.nlargest(10).sum() / total_medals.sum()
            
            concentration[f"{start}-{end}"] = {
                'gini_coefficient': gini,
                'top_10_share': top_10_share,
                'total_countries': len(total_medals)
            }
            
        return concentration

    def _calculate_gini(self, array: np.ndarray) -> float:
        """计算基尼系数"""
        array = array.flatten()
        if len(array) == 0:
            return 0
        array = np.sort(array)
        index = np.arange(1, len(array) + 1)
        n = len(array)
        return ((2 * index - n - 1) * array).sum() / (n * array.sum())
    def _identify_emerging_countries(self, data: pd.DataFrame, recent_years: int = 12) -> List[Dict]:
        """改进的新兴国家识别算法，使用复合指标"""
        historical_nocs = {'URS', 'GDR', 'FRG', 'EUN', 'YUG', 'TCH', 'ROC'}
        data = data[~data['NOC'].isin(historical_nocs)].copy()
        
        def calculate_growth_score(early_medals: float, recent_medals: float) -> float:
            """使用对数增长率避免除零问题"""
            if early_medals == 0:
                early_medals = 0.5  # Laplace平滑
            return np.log((recent_medals + 1) / (early_medals + 1))
        
        def calculate_momentum_score(country_data: pd.DataFrame) -> float:
            """计算动量得分"""
            if len(country_data) < 3:
                return 0
            
            # 使用指数加权移动平均
            weights = np.exp(np.linspace(-1, 0, len(country_data)))
            weighted_avg = np.average(country_data['Total'], weights=weights)
            return weighted_avg
        
        emerging_metrics = []
        max_year = data['Year'].max()
        
        for noc in data['NOC'].unique():
            country_data = data[data['NOC'] == noc]
            
            # 分割时期
            recent_data = country_data[country_data['Year'] >= (max_year - recent_years)]
            early_data = country_data[country_data['Year'] < (max_year - recent_years)]
            
            if len(recent_data) < 2 or len(early_data) < 2:
                continue
                
            # 计算复合指标
            growth_score = calculate_growth_score(
                early_data['Total'].mean(),
                recent_data['Total'].mean()
            )
            
            momentum_score = calculate_momentum_score(recent_data)
            
            # 计算稳定性
            stability = 1 / (1 + recent_data['Total'].std())
            
            # 综合得分
            final_score = (
                growth_score * 0.4 +
                momentum_score * 0.4 +
                stability * 0.2
            )
            
            if final_score > 0:
                emerging_metrics.append({
                    'NOC': noc,
                    'growth_rate': float(growth_score),
                    'momentum': float(momentum_score),
                    'stability': float(stability),
                    'final_score': float(final_score),
                    'recent_medals': float(recent_data['Total'].mean()),
                    'early_medals': float(early_data['Total'].mean())
                })
        
        # 按综合得分排序
        return sorted(emerging_metrics, key=lambda x: x['final_score'], reverse=True)[:10]
    # def _identify_emerging_countries(self, data: pd.DataFrame, recent_years: int = 12) -> List[Dict]:
    #     """改进的新兴国家识别算法"""
    #     # 数据预处理
    #     historical_nocs = {'URS', 'GDR', 'FRG', 'EUN', 'YUG', 'TCH', 'ROC'}
    #     data = data[~data['NOC'].isin(historical_nocs)].copy()
        
    #     # 计算时期表现
    #     max_year = data['Year'].max()
    #     recent_data = data[data['Year'] >= (max_year - recent_years)]
    #     early_data = data[data['Year'] < (max_year - recent_years)]
        
    #     # 计算平均值
    #     recent_avg = recent_data.groupby('NOC')['Total'].mean()
    #     early_avg = early_data.groupby('NOC')['Total'].mean()
        
    #     # 计算增长
    #     growth_df = pd.DataFrame({
    #         'recent': recent_avg,
    #         'early': early_avg
    #     }).fillna(0)
        
    #     # 改进的增长率计算
    #     growth_df['growth_rate'] = (
    #         (growth_df['recent'] - growth_df['early']) / 
    #         (growth_df['early'].replace(0, 0.1))  # 避免除以0
    #     ) * 100
        
    #     # 限制极端值
    #     growth_df['growth_rate'] = growth_df['growth_rate'].clip(-1000, 1000)
        
    #     # 筛选条件
    #     emerging = growth_df[
    #         (growth_df['growth_rate'] > 20) &  # 显著增长
    #         (growth_df['recent'] >= 3) &       # 当前有一定规模
    #         (growth_df['recent'] > growth_df['early'])  # 确实在增长
    #     ].sort_values('growth_rate', ascending=False)
        
    #     return [
    #         {
    #             'NOC': noc,
    #             'growth_rate': float(emerging.loc[noc, 'growth_rate']),
    #             'recent_medals': float(emerging.loc[noc, 'recent']),
    #             'early_medals': float(growth_df.loc[noc, 'early'])
    #         }
    #         for noc in emerging.index[:10]
    #     ]

    def _identify_declining_countries(self, data: pd.DataFrame, recent_years: int = 12) -> List[Dict]:
        """改进的衰退国家识别"""
        # 过滤历史国家
        historical_nocs = {'URS', 'GDR', 'FRG', 'EUN', 'YUG', 'TCH', 'ROC', 
                        'Soviet Union', 'East Germany', 'West Germany', 'Unified Team'}
        
        data = data[~data['NOC'].isin(historical_nocs)].copy()
        
        # 计算各个时期的表现
        recent_data = data[data['Year'] >= data['Year'].max() - recent_years]
        early_data = data[data['Year'] < data['Year'].max() - recent_years]
        
        # 使用更复杂的衰退指标
        decline_metrics = []
        
        for noc in data['NOC'].unique():
            noc_recent = recent_data[recent_data['NOC'] == noc]
            noc_early = early_data[early_data['NOC'] == noc]
            
            if len(noc_recent) < 2 or len(noc_early) < 2:
                continue
            
            # 计算多个维度的衰退指标
            medal_decline = (noc_early['Total'].mean() - noc_recent['Total'].mean()) / (noc_early['Total'].mean() + 1)
            consistency_decline = noc_recent['Total'].std() / (noc_recent['Total'].mean() + 1) - \
                                noc_early['Total'].std() / (noc_early['Total'].mean() + 1)
            peak_decline = noc_early['Total'].max() - noc_recent['Total'].max()
            
            # 仅考虑最近仍有参与的国家
            if noc_recent['Year'].max() < data['Year'].max() - 4:
                continue
                
            # 综合衰退分数
            decline_score = (medal_decline * 0.5 + 
                            consistency_decline * 0.3 + 
                            (peak_decline / (noc_early['Total'].max() + 1)) * 0.2)
            
            if decline_score > 0.2:  # 显著衰退阈值
                decline_metrics.append({
                    'NOC': noc,
                    'decline_rate': float(decline_score),
                    'recent_medals': float(noc_recent['Total'].mean()),
                    'early_medals': float(noc_early['Total'].mean()),
                    'peak_medals': float(noc_early['Total'].max()),
                    'last_appearance': int(noc_recent['Year'].max())
                })
        
        return sorted(decline_metrics, key=lambda x: x['decline_rate'], reverse=True)[:10]

    def _create_region_mapping(self) -> Dict[str, str]:
        """完整的国家-地区映射"""
        return {
            # 北美洲
            'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
            'CUB': 'North America', 'JAM': 'North America', 'PUR': 'North America',
            'DOM': 'North America', 'TTO': 'North America', 'BAH': 'North America',
            'BAR': 'North America', 'CRC': 'North America', 'GUA': 'North America',
            
            # 南美洲
            'BRA': 'South America', 'ARG': 'South America', 'CHI': 'South America',
            'COL': 'South America', 'VEN': 'South America', 'ECU': 'South America',
            'PER': 'South America', 'URU': 'South America', 'PAR': 'South America',
            'BOL': 'South America', 'SUR': 'South America', 'GUY': 'South America',
            
            # 欧洲
            'GBR': 'Europe', 'FRA': 'Europe', 'GER': 'Europe', 'ITA': 'Europe',
            'ESP': 'Europe', 'NED': 'Europe', 'SWE': 'Europe', 'POL': 'Europe',
            'HUN': 'Europe', 'ROU': 'Europe', 'CZE': 'Europe', 'GRE': 'Europe',
            'BUL': 'Europe', 'BEL': 'Europe', 'AUT': 'Europe', 'FIN': 'Europe',
            'NOR': 'Europe', 'SUI': 'Europe', 'DEN': 'Europe', 'CRO': 'Europe',
            'SRB': 'Europe', 'UKR': 'Europe', 'BLR': 'Europe', 'SVK': 'Europe',
            'IRL': 'Europe', 'POR': 'Europe', 'LAT': 'Europe', 'EST': 'Europe',
            'LTU': 'Europe', 'MDA': 'Europe', 'RUS': 'Europe', 'ALB': 'Europe',
            
            # 亚洲
            'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia', 'PRK': 'Asia',
            'MNG': 'Asia', 'KAZ': 'Asia', 'UZB': 'Asia', 'KGZ': 'Asia',
            'TJK': 'Asia', 'TKM': 'Asia', 'IND': 'Asia', 'PAK': 'Asia',
            'BAN': 'Asia', 'SRI': 'Asia', 'NEP': 'Asia', 'THA': 'Asia',
            'VNM': 'Asia', 'IDN': 'Asia', 'MAS': 'Asia', 'SGP': 'Asia',
            'PHI': 'Asia', 'MYA': 'Asia', 'LAO': 'Asia', 'BRU': 'Asia',
            'IRN': 'Asia', 'IRQ': 'Asia', 'SAU': 'Asia', 'QAT': 'Asia',
            'BHR': 'Asia', 'KWT': 'Asia', 'UAE': 'Asia', 'YEM': 'Asia',
            'OMA': 'Asia', 'SYR': 'Asia', 'LBN': 'Asia', 'JOR': 'Asia',
            'AFG': 'Asia', 'TLS': 'Asia',
            
            # 大洋洲
            'AUS': 'Oceania', 'NZL': 'Oceania', 'FIJ': 'Oceania', 'PNG': 'Oceania',
            'SAM': 'Oceania', 'SOL': 'Oceania', 'VAN': 'Oceania', 'TON': 'Oceania',
            
            # 非洲
            'RSA': 'Africa', 'NGR': 'Africa', 'KEN': 'Africa', 'ETH': 'Africa',
            'EGY': 'Africa', 'MAR': 'Africa', 'ALG': 'Africa', 'TUN': 'Africa',
            'UGA': 'Africa', 'GHA': 'Africa', 'CMR': 'Africa', 'ZIM': 'Africa',
            'CIV': 'Africa', 'SEN': 'Africa', 'GAB': 'Africa', 'COD': 'Africa',
            'NAM': 'Africa', 'BOT': 'Africa', 'MAD': 'Africa', 'MLI': 'Africa',
            'BDI': 'Africa', 'RWA': 'Africa', 'SUD': 'Africa', 'TAN': 'Africa',
            'MOZ': 'Africa', 'SEY': 'Africa', 'MRI': 'Africa', 'BEN': 'Africa',
            'TOG': 'Africa', 'LBA': 'Africa', 'DJI': 'Africa', 'GAM': 'Africa',
            'SLE': 'Africa', 'SOM': 'Africa', 'ZAM': 'Africa',
            
            # 历史国家/特殊地区
            'URS': 'Europe', 'GDR': 'Europe', 'FRG': 'Europe', 'TCH': 'Europe',
            'YUG': 'Europe', 'EUN': 'Europe', 'ROC': 'Europe'
        }

    def _analyze_regional_patterns(self, data: pd.DataFrame) -> Dict:
        """增强版区域分析，包含历史演变和主导项目分析"""
        region_patterns = {}
        regions = ['Europe', 'Asia', 'North America', 'South America', 'Africa', 'Oceania']
        
        # 确保区域映射
        data['Region'] = data['NOC'].map(self._create_region_mapping())
        data = data[data['Region'].isin(regions)]
        
        # 定义分析时期
        current_year = data['Year'].max()
        periods = [
            (1896, 1936, 'Early Era'),
            (1948, 1988, 'Cold War Era'),
            (1992, 2012, 'Post-Cold War'),
            (2016, current_year, 'Recent Era')
        ]
        
        for region in regions:
            region_data = data[data['Region'] == region]
            if len(region_data) < 2:
                continue
            
            # 初始化区域分析
            analysis = {
                'historical_evolution': {},
                'recent_performance': {},
                'dominant_sports': {},
                'emerging_trends': {},
                'key_metrics': {}
            }
            
            # 1. 历史演变分析
            for start, end, era in periods:
                era_data = region_data[(region_data['Year'] >= start) & (region_data['Year'] <= end)]
                if len(era_data) > 0:
                    total_medals = era_data['Total'].sum()
                    global_medals = data[(data['Year'] >= start) & (data['Year'] <= end)]['Total'].sum()
                    
                    analysis['historical_evolution'][era] = {
                        'medal_share': float(total_medals / global_medals if global_medals > 0 else 0),
                        'total_medals': int(total_medals),
                        'participating_countries': len(era_data['NOC'].unique()),
                        'top_performers': era_data.groupby('NOC')['Total'].sum().nlargest(3).to_dict()
                    }
            
            # 2. 近期表现分析
            recent_data = region_data[region_data['Year'] > current_year - 12]
            if len(recent_data) > 0:
                analysis['recent_performance'] = {
                    'average_medals': float(recent_data.groupby('Year')['Total'].sum().mean()),
                    'trend': self._calculate_robust_trend(recent_data.groupby('Year')['Total'].sum()),
                    'stability': float(1 / (1 + recent_data.groupby('Year')['Total'].sum().std())),
                    'top_countries': recent_data.groupby('NOC')['Total'].sum().nlargest(5).to_dict()
                }
            
            # 3. 主导项目分析
            if 'Sport' in region_data.columns:
                recent_sports = recent_data.groupby('Sport')['Total'].sum()
                historical_sports = region_data[region_data['Year'] <= current_year - 12].groupby('Sport')['Total'].sum()
                
                analysis['dominant_sports'] = {
                    'current_strengths': recent_sports.nlargest(5).to_dict(),
                    'historical_strengths': historical_sports.nlargest(5).to_dict(),
                    'emerging_sports': {
                        sport: float(recent_sports[sport])
                        for sport in recent_sports.index
                        if sport in historical_sports.index and
                        recent_sports[sport] > historical_sports[sport] * 1.5
                    }
                }
            
            # 4. 趋势分析
            analysis['emerging_trends'] = {
                'growth_rate': self._calculate_region_growth(region_data),
                'diversification': len(recent_data['Sport'].unique()) / len(data['Sport'].unique()) if 'Sport' in data.columns else 0,
                'new_competitors': len(set(recent_data['NOC']) - set(region_data[region_data['Year'] <= current_year - 12]['NOC']))
            }
            
            # 5. 关键指标
            analysis['key_metrics'] = {
                'medals_per_country': float(recent_data['Total'].sum() / len(recent_data['NOC'].unique())),
                'consistency_score': self._calculate_consistency_score(region_data),
                'development_index': self._calculate_development_index(region_data)
            }
            
            region_patterns[region] = analysis
        
        return region_patterns
    def _calculate_development_index(self, data: pd.DataFrame) -> float:
        """计算地区发展指数"""
        if len(data) < 2:
            return 0.0
        
        try:
            recent_years = data['Year'].max() - 8
            recent_data = data[data['Year'] >= recent_years]
            historical_data = data[data['Year'] < recent_years]
            
            # 计算各项指标
            medal_growth = (recent_data['Total'].mean() / (historical_data['Total'].mean() + 1) - 1)
            country_growth = len(recent_data['NOC'].unique()) / (len(historical_data['NOC'].unique()) + 1) - 1
            consistency = 1 / (1 + recent_data.groupby('Year')['Total'].sum().std())
            
            # 综合评分
            development_score = (
                0.4 * medal_growth +
                0.3 * country_growth +
                0.3 * consistency
            )
            
            return float(max(0, min(1, (development_score + 1) / 2)))
        except Exception as e:
            print(f"开发指数计算错误: {str(e)}")
            return 0.0
    def _calculate_region_stability(self, region_data: pd.DataFrame) -> Dict:
        """Calculate region's performance stability"""
        recent_years = region_data['Year'].max() - 8
        recent_data = region_data[region_data['Year'] > recent_years]
        
        if len(recent_data) < 2:
            return {'coefficient_variation': 0.0, 'trend_consistency': 0.0}
        
        # Calculate coefficient of variation
        cv = float(recent_data.groupby('Year')['Total'].sum().std() / 
                recent_data.groupby('Year')['Total'].sum().mean())
        
        # Calculate trend consistency
        yearly_medals = recent_data.groupby('Year')['Total'].sum()
        trend = np.polyfit(range(len(yearly_medals)), yearly_medals.values, 1)[0]
        residuals = np.abs(yearly_medals.values - np.polyval(
            np.polyfit(range(len(yearly_medals)), yearly_medals.values, 1),
            range(len(yearly_medals))
        ))
        trend_consistency = float(1 - np.mean(residuals) / yearly_medals.mean())
        
        return {
            'coefficient_variation': cv,
            'trend_consistency': trend_consistency
        }
    def _get_region_dominant_sports(self, region_data: pd.DataFrame) -> Dict:
        """Calculate region's dominant sports"""
        if 'Sport' not in region_data.columns:
            return {'top_sports': {}, 'emerging_sports': {}}
        
        recent_cutoff = region_data['Year'].max() - 8
        
        # Calculate sport dominance
        recent_sports = region_data[
            region_data['Year'] > recent_cutoff
        ].groupby('Sport')['Total'].sum()
        
        historical_sports = region_data[
            region_data['Year'] <= recent_cutoff
        ].groupby('Sport')['Total'].sum()
        
        # Identify emerging sports
        emerging_sports = {}
        for sport in recent_sports.index:
            recent_medals = recent_sports.get(sport, 0)
            historical_medals = historical_sports.get(sport, 0)
            if recent_medals > historical_medals * 1.5:  # 50% growth threshold
                emerging_sports[sport] = float(recent_medals)
        
        return {
            'top_sports': recent_sports.nlargest(5).to_dict(),
            'emerging_sports': emerging_sports
        }
    def _calculate_trend_score(self, data: pd.DataFrame, noc: str) -> float:
        """
        Enhanced trend score calculation with dynamic weighting and host effect
        """
        country_data = data[data['NOC'] == noc].copy()
        if len(country_data) < 4:
            return 0.0
            
        try:
            # Split data into periods
            max_year = country_data['Year'].max()
            recent_cutoff = max_year - 8  # Last 2 Olympics
            mid_cutoff = max_year - 16    # Previous 2 Olympics
            
            recent_data = country_data[country_data['Year'] > recent_cutoff]
            mid_data = country_data[(country_data['Year'] <= recent_cutoff) & 
                                (country_data['Year'] > mid_cutoff)]
            early_data = country_data[country_data['Year'] <= mid_cutoff]
            
            # Calculate period averages
            recent_avg = recent_data['Total'].mean() if len(recent_data) > 0 else 0
            mid_avg = mid_data['Total'].mean() if len(mid_data) > 0 else 0
            early_avg = early_data['Total'].mean() if len(early_data) > 0 else 0
            
            # Calculate growth rates
            recent_growth = (recent_avg - mid_avg) / (mid_avg + 1)
            historical_growth = (mid_avg - early_avg) / (early_avg + 1)
            
            # Adjust for host effect
            host_years = {2024: 'FRA', 2021: 'JPN', 2016: 'BRA', 2012: 'GBR'}
            host_bonus = 0
            for year, host in host_years.items():
                if noc == host:
                    years_until = max_year - year
                    if years_until >= 0:
                        host_bonus = max(0, 0.2 * (1 - years_until/8))  # Decay over 8 years
            
            # Calculate consistency
            recent_consistency = 1 / (1 + recent_data['Total'].std()) if len(recent_data) > 1 else 0
            
            # Combine metrics with weights
            trend_score = (
                0.5 * recent_growth +      # Recent performance
                0.2 * historical_growth +   # Historical trend
                0.2 * recent_consistency +  # Stability
                0.1 * host_bonus           # Host country effect
            )
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (trend_score + 1) / 2))
            
        except Exception as e:
            print(f"Error calculating trend score for {noc}: {str(e)}")
            return 0.0
    def _calculate_robust_trend(self, series: pd.Series) -> float:
        """使用Theil-Sen回归计算稳健趋势"""
        if len(series) < 2:
            return 0.0
        years = series.index.values.reshape(-1, 1)
        medals = series.values
        slope, _, _, _ = stats.theilslopes(medals, years)
        return float(slope)
    def _calculate_region_trend(self, region_data: pd.DataFrame) -> float:
        """使用Huber回归计算区域趋势"""
        from sklearn.linear_model import HuberRegressor
        X = region_data[['Year']].values
        y = region_data['Total'].values
        model = HuberRegressor().fit(X, y)
        return float(model.coef_[0])
    def _analyze_sport_diversity(self, data: pd.DataFrame) -> Dict:
        """改进的运动项目多样性分析，增加错误检查"""
        try:
            diversity = {}
            
            # 数据验证
            if 'Sport' not in data.columns:
                raise ValueError("Sport列不存在于数据中")
                
            if 'NOC' not in data.columns:
                raise ValueError("NOC列不存在于数据中")
                
            # 打印整体统计信息
            print("\n运动项目多样性分析诊断:")
            print(f"总记录数: {len(data)}")
            print(f"唯一国家数: {data['NOC'].nunique()}")
            print(f"唯一运动项目数: {data['Sport'].nunique()}")
            
            # 计算每个国家在不同项目上的分布
            country_sports = data.groupby('NOC')['Sport'].nunique()
            print(f"\n国家运动项目分布:")
            print(f"最大值: {country_sports.max()}")
            print(f"最小值: {country_sports.min()}")
            print(f"平均值: {country_sports.mean():.2f}")
            print(f"中位数: {country_sports.median()}")
            
            # 计算多样性指标
            diversity['overall'] = {
                'avg_sports': float(country_sports.mean()),
                'max_sports': int(country_sports.max()),
                'min_sports': int(country_sports.min()),
                'median_sports': float(country_sports.median()),
                'total_sports': len(data['Sport'].unique())
            }
            
            # 识别专注型和多样化型国家
            q25, q75 = country_sports.quantile([0.25, 0.75])
            specialized = country_sports[country_sports < q25]
            diversified = country_sports[country_sports > q75]
            
            print(f"\n专注型国家(低于25分位): {len(specialized)}")
            print(f"多样化国家(高于75分位): {len(diversified)}")
            
            diversity['specialized'] = specialized.to_dict()
            diversity['diversified'] = diversified.to_dict()
            
            # 时间趋势分析
            if 'Year' in data.columns:
                recent_years = data['Year'].max() - 8
                recent_data = data[data['Year'] >= recent_years]
                recent_sports = recent_data.groupby('NOC')['Sport'].nunique()
                
                diversity['recent_trends'] = {
                    'avg_sports_recent': float(recent_sports.mean()),
                    'max_sports_recent': int(recent_sports.max()),
                    'countries_increased': len(recent_sports[recent_sports > country_sports])
                }
            
            return diversity
            
        except Exception as e:
            print(f"错误: 分析运动项目多样性时发生异常: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                'overall': {'avg_sports': 0, 'max_sports': 0, 'min_sports': 0},
                'specialized': {},
                'diversified': {}
            }

    def generate_country_insights(self, medals_df: pd.DataFrame, athletes_df: pd.DataFrame) -> Dict[str, CountryInsight]:
        insights = {}
        
        medals_df = medals_df.copy()
        athletes_df = athletes_df.copy()

        # 预处理：清理NOC代码中的额外空格
        medals_df['NOC'] = medals_df['NOC'].str.strip()
        
        # NOC标准化映射
        noc_mapping = {
            'UNITED STATES': 'USA', 'GREAT BRITAIN': 'GBR', 'SOVIET UNION': 'URS',
            'FRANCE': 'FRA', 'CHINA': 'CHN', 'GERMANY': 'GER', 'ITALY': 'ITA',
            'AUSTRALIA': 'AUS', 'JAPAN': 'JPN', 'HUNGARY': 'HUN', 'SWEDEN': 'SWE',
            'RUSSIA': 'RUS', 'EAST GERMANY': 'GDR', 'NETHERLANDS': 'NED',
            'CANADA': 'CAN', 'SOUTH KOREA': 'KOR', 'ROMANIA': 'ROU', 'POLAND': 'POL',
            'FINLAND': 'FIN', 'CUBA': 'CUB', 'BULGARIA': 'BUL', 'SWITZERLAND': 'SUI',
            'WEST GERMANY': 'FRG', 'DENMARK': 'DEN', 'SPAIN': 'ESP', 'NORWAY': 'NOR',
            'BRAZIL': 'BRA', 'BELGIUM': 'BEL', 'NEW ZEALAND': 'NZL'
        }
        
        medals_df['NOC'] = medals_df['NOC'].apply(lambda x: noc_mapping.get(x.upper(), x))
        
        main_countries = medals_df.groupby('NOC')['Total'].sum().nlargest(30).index
        
        for noc in main_countries:
            trend_score = self._calculate_trend_score(medals_df, noc)
            stability_score = self._calculate_stability_score(medals_df, noc)
            diversity_score = self._calculate_diversity_score(athletes_df, noc)
            
            key_findings = self._generate_key_findings(
                medals_df, athletes_df, noc,
                trend_score, stability_score, diversity_score
            )
            
            recommendations = self._generate_recommendations(
                key_findings, trend_score, stability_score, diversity_score
            )
            
            insights[noc] = CountryInsight(
                noc=noc,
                trend_score=trend_score,
                stability_score=stability_score,
                diversity_score=diversity_score,
                key_findings=key_findings,
                recommendations=recommendations
            )
        
        return insights
    def _calculate_trend_score(self, data: pd.DataFrame, noc: str) -> float:
        """
        Calculate trend score using weighted recent performance and robust regression
        """
        country_data = data[data['NOC'] == noc].copy()
        if len(country_data) < 4:
            return 0.0
            
        try:
            # Split data into recent and historical periods
            max_year = country_data['Year'].max()
            recent_cutoff = max_year - 12  # Last 3 Olympics
            
            recent_data = country_data[country_data['Year'] > recent_cutoff]
            historical_data = country_data[country_data['Year'] <= recent_cutoff]
            
            # Calculate trends for both periods using Theil-Sen estimator
            def calc_period_trend(df):
                if len(df) < 2:
                    return 0
                years = (df['Year'] - df['Year'].min()).values
                medals = df['Total'].values
                return stats.theilslopes(medals, years)[0]
                
            recent_trend = calc_period_trend(recent_data)
            historical_trend = calc_period_trend(historical_data)
            
            # Calculate relative performance
            recent_avg = recent_data['Total'].mean() if len(recent_data) > 0 else 0
            historical_avg = historical_data['Total'].mean() if len(historical_data) > 0 else 0
            relative_change = (recent_avg - historical_avg) / (historical_avg + 1)
            
            # Combine metrics with weights
            trend_score = (
                0.5 * (recent_trend / (abs(recent_trend) + 1)) +  # Recent trend (normalized)
                0.3 * (historical_trend / (abs(historical_trend) + 1)) +  # Historical trend
                0.2 * (relative_change)  # Overall improvement
            )
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (trend_score + 1) / 2))
            
        except Exception as e:
            print(f"Error calculating trend score for {noc}: {str(e)}")
            return 0.0

    def _calculate_stability_score(self, data: pd.DataFrame, noc: str) -> float:
        """计算国家表现的稳定性得分"""
        country_data = data[data['NOC'] == noc]
        if len(country_data) < 2:
            return 0.0
            
        # 计算变异系数（标准差/平均值）
        cv = country_data['Total'].std() / (country_data['Total'].mean() + 1e-6)
        
        # 转换为稳定性得分（越稳定越接近1）
        return 1 / (1 + cv)

    def _calculate_diversity_score(self, data: pd.DataFrame, noc: str) -> float:
        """改进的多样性得分计算"""
        try:
            # 规范化NOC处理
            standardized_noc = noc.strip().upper()
            country_data = data[data['NOC'] == standardized_noc].copy()
            
            if len(country_data) == 0:
                print(f"警告: {noc} (标准化后: {standardized_noc}) 没有运动项目数据")
                # 尝试模糊匹配
                similar_nocs = data['NOC'].unique()
                print(f"数据中存在的相似NOC: {[n for n in similar_nocs if n.startswith(standardized_noc[:3])]}")
                return 0.0
                
            # 基本验证
            if 'Sport' not in country_data.columns:
                print(f"错误: Sport列不存在")
                return 0.0
                
            # 数据统计
            total_sports = len(data['Sport'].unique())
            country_sports = len(country_data['Sport'].unique())
            
            # 1. 规模得分 (0-0.4)
            scale_score = 0.4 * (country_sports / total_sports) if total_sports > 0 else 0
            
            # 2. 均衡度得分 (0-0.3)
            sport_counts = country_data['Sport'].value_counts()
            if len(sport_counts) > 1:
                probs = sport_counts / sport_counts.sum()
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(len(sport_counts))
                balance_score = 0.3 * (entropy / max_entropy)
            else:
                balance_score = 0.0
            
            # 3. 参与度得分 (0-0.3)
            recent_years = data['Year'].max() - 8
            recent_data = country_data[country_data['Year'] >= recent_years]
            participation_rate = len(recent_data['Sport'].unique()) / max(country_sports, 1)
            participation_score = 0.3 * participation_rate
            
            final_score = scale_score + balance_score + participation_score
            
            return round(min(1.0, max(0.0, final_score)), 2)
            
        except Exception as e:
            print(f"错误: 计算{noc}的多样性得分时发生异常: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return 0.0

    def _generate_key_findings(self, 
                             medals_df: pd.DataFrame, 
                             athletes_df: pd.DataFrame, 
                             noc: str,
                             trend_score: float,
                             stability_score: float,
                             diversity_score: float) -> List[str]:
        """生成关键发现"""
        findings = []
        
        # 分析趋势
        if trend_score > 0.7:
            findings.append("显示出强劲的上升势头")
        elif trend_score < 0.3:
            findings.append("表现呈现下降趋势")
        
        # 分析稳定性
        if stability_score > 0.7:
            findings.append("表现非常稳定")
        elif stability_score < 0.3:
            findings.append("表现波动较大")
        
        # 分析多样性
        if diversity_score > 0.7:
            findings.append("具有良好的项目多样性")
        elif diversity_score < 0.3:
            findings.append("项目集中度较高")
        
        return findings

    def _generate_recommendations(self, 
                                findings: List[str],
                                trend_score: float,
                                stability_score: float,
                                diversity_score: float,
                                noc: str = None,
                                medals_df: pd.DataFrame = None) -> List[str]:
        """
        Generate targeted recommendations based on comprehensive analysis
        """
        recommendations = []
        
        # Get country-specific metrics if available
        dominant_sports = []
        emerging_sports = []
        if noc and medals_df is not None:
            country_data = medals_df[medals_df['NOC'] == noc]
            if 'Sport' in country_data.columns:
                recent_data = country_data[
                    country_data['Year'] >= country_data['Year'].max() - 8
                ]
                dominant_sports = recent_data.groupby('Sport')['Total'].sum().nlargest(3).index.tolist()
                
                historical_data = country_data[
                    country_data['Year'] < country_data['Year'].max() - 8
                ]
                emerging_sports = [
                    sport for sport in recent_data['Sport'].unique()
                    if recent_data[recent_data['Sport'] == sport]['Total'].sum() >
                    1.5 * historical_data[historical_data['Sport'] == sport]['Total'].sum()
                ]
        
        # Trend-based recommendations
        if trend_score < 0.3:
            if dominant_sports:
                recommendations.append(
                    f"重点投入{', '.join(dominant_sports[:2])}等优势项目，"
                    "制定中长期人才培养计划"
                )
            else:
                recommendations.append("加大重点项目投入，制定中长期人才培养计划")
        elif trend_score < 0.6:
            if emerging_sports:
                recommendations.append(
                    f"在保持现有优势的同时，重点发展{', '.join(emerging_sports[:2])}等新兴项目"
                )
            else:
                recommendations.append("保持现有优势项目投入，同时开拓新的潜力项目")
        else:
            if dominant_sports:
                recommendations.append(
                    f"巩固{', '.join(dominant_sports)}等优势项目的领先地位，"
                    "建立可持续的竞技体系"
                )
            else:
                recommendations.append("巩固优势项目领先地位，建立可持续的竞技体系")
        
        # Stability-based recommendations
        if stability_score < 0.4:
            recommendations.append(
                "加强后备人才梯队建设，建立科学的选材和培养体系"
            )
        elif stability_score < 0.7:
            recommendations.append(
                "优化训练体系，提高竞技水平的稳定性和可持续性"
            )
        else:
            recommendations.append(
                "完善人才储备机制，保持成绩稳定性和竞争优势"
            )
        
        # Diversity-based recommendations
        if diversity_score < 0.3:
            if dominant_sports:
                recommendations.append(
                    f"在{dominant_sports[0]}项目基础上，发展相关联项目，培养复合型人才"
                )
            else:
                recommendations.append("拓展优势项目相关联项目，培养复合型人才")
        elif diversity_score < 0.6:
            recommendations.append(
                "在保持优势项目的同时，积极开发具有突破潜力的新项目"
            )
        else:
            recommendations.append(
                "优化资源分配策略，维持项目多样性优势"
            )
        
        return recommendations

    def generate_report(self, trends: Dict, insights: Dict[str, CountryInsight]) -> str:
        """生成分析报告"""
        report = []
        
        # 1. 总体趋势
        report.append("1. 奥运会奖牌总体趋势分析")
        report.append("-" * 50)
        
        # 添加整体趋势分析
        overall = trends.get('overall', {})
        report.append(f"\n参与国家数量: {overall.get('total_countries', 'N/A')}")
        
        # 添加奖牌集中度分析
        concentration = overall.get('medals_concentration', {})
        for period, stats in concentration.items():
            report.append(f"\n{period}时期:")
            report.append(f"  - 基尼系数: {stats['gini_coefficient']:.3f}")
            report.append(f"  - 前10国家占比: {stats['top_10_share']*100:.1f}%")
        
        # 2. 新兴与衰退趋势
        report.append("\n\n2. 新兴与衰退国家分析")
        report.append("-" * 50)
        
        # 新兴国家
        emerging = overall.get('emerging_countries', [])
        report.append("\n新兴奥运强国:")
        for country in emerging[:5]:
            report.append(
                f"  - {country['NOC']}: 增长率 {country['growth_rate']*100:.1f}%, "
                f"近期平均 {country['recent_medals']:.1f} 枚奖牌"
            )
        
        # 衰退国家
        declining = overall.get('declining_countries', [])
        report.append("\n实力下降国家:")
        for country in declining[:5]:
            report.append(
                f"  - {country['NOC']}: 下降率 {country['decline_rate']*100:.1f}%, "
                f"近期平均 {country['recent_medals']:.1f} 枚奖牌"
            )
        
        # 3. 区域分析
        report.append("\n\n3. 区域性分析")
        report.append("-" * 50)
        
        regional = trends.get('regional', {})
        for region, stats in regional.items():
            report.append(f"\n{region}:")
            report.append(f"  - 趋势系数: {stats['trend']:.2f}")
            report.append(f"  - 波动性: {stats['volatility']:.2f}")
            report.append(f"  - 奖牌中位数: {stats['medal_median']:.1f}")
            report.append(f"  - 近期份额: {stats['recent_share']*100:.1f}%")
        
        # 4. 国家深度分析
        report.append("\n\n4. 国家深度分析")
        report.append("-" * 50)
        
        for noc, insight in list(insights.items())[:10]:  # 展示前10个国家
            report.append(f"\n{noc}分析:")
            report.append(f"  趋势得分: {insight.trend_score:.2f}")
            report.append(f"  稳定性得分: {insight.stability_score:.2f}")
            report.append(f"  多样性得分: {insight.diversity_score:.2f}")
            report.append("  主要发现:")
            for finding in insight.key_findings:
                report.append(f"    - {finding}")
            report.append("  建议:")
            for recommendation in insight.recommendations:
                report.append(f"    - {recommendation}")
        
        return "\n".join(report)

def main():
    console = Console()
    
    try:
        # 初始化分析器
        analyzer = OlympicMedalInsightAnalyzer()
        
        # 加载数据
        console.print("[bold cyan]加载数据...[/bold cyan]")
        medals_df, athletes_df, programs_df = analyzer.load_and_prepare_data()
        
        # 分析趋势
        console.print("[bold cyan]分析奖牌趋势...[/bold cyan]")
        # 在 main() 函数中
        trends = analyzer.analyze_medal_trends(medals_df, athletes_df)
        analyzer.visualizer.plot_medal_trends(trends)
        # 生成国家洞察
        console.print("[bold cyan]生成国家洞察...[/bold cyan]")
        insights = analyzer.generate_country_insights(medals_df, athletes_df)
        analyzer.visualizer.plot_country_insights(insights)
        # 生成报告
        report = analyzer.generate_report(trends, insights)
        
        # 保存报告
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "olympic_medal_insights_report.txt", "w", encoding='utf-8') as f:
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