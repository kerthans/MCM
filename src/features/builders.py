# src/features/builders.py
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from rich.progress import track
from rich.console import Console
from src.utils.logger import log_info, log_success

class FeatureBuilder:
    def __init__(self, medals_df, athletes_df, hosts_df, programs_df):
        self.medals_df = medals_df
        self.athletes_df = athletes_df
        self.hosts_df = hosts_df
        self.programs_df = programs_df
        self.console = Console()
        
    def build_all_features(self):
        """构建所有特征"""
        features = {}
        
        with self.console.status("[bold green]Building features...") as status:
            # 时间序列特征
            features['time_series'] = self._build_time_series_features()
            status.update("[bold green]Time series features built")
            
            # 国家特征
            features['country'] = self._build_country_features()
            status.update("[bold green]Country features built")
            
            # 项目特征
            features['sport'] = self._build_sport_features()
            status.update("[bold green]Sport features built")
            
            # 主办国特征
            features['host'] = self._build_host_features()
            status.update("[bold green]Host features built")
            
            # 教练效应特征
            features['coach'] = self._build_coach_features()
            status.update("[bold green]Coach features built")
            
        return features
    
    def _build_time_series_features(self):
        """构建时间序列特征"""
        # 按国家和年份分组的奖牌数据
        time_series = self.medals_df.pivot(index='Year', columns='NOC', values='Total').fillna(0)
        
        # 使用tsfresh提取特征
        fc_parameters = MinimalFCParameters()
        ts_features = extract_features(
            time_series.reset_index(),
            column_id='Year',
            column_sort='Year',
            default_fc_parameters=fc_parameters
        )
        
        return ts_features
    
    def _build_country_features(self):
        """构建国家相关特征"""
        country_features = pd.DataFrame()
        
        # 历史表现统计
        history_stats = self.medals_df.groupby('NOC').agg({
            'Gold': ['mean', 'std', 'max'],
            'Total': ['mean', 'std', 'max']
        }).round(2)
        
        # 近期表现趋势
        recent_years = self.medals_df[self.medals_df['Year'] >= 2000]
        recent_trends = recent_years.groupby('NOC').agg({
            'Gold': 'mean',
            'Total': 'mean'
        }).add_prefix('recent_')
        
        # 项目多样性
        sport_diversity = self.athletes_df.groupby('NOC')['Sport'].nunique().rename('sport_count')
        
        country_features = pd.concat([history_stats, recent_trends, sport_diversity], axis=1)
        return country_features
    
    def _build_sport_features(self):
        """构建项目相关特征"""
        sport_features = pd.DataFrame()
        
        # 项目成功率
        medal_counts = self.athletes_df[self.athletes_df['Medal'].notna()].groupby(['NOC', 'Sport']).size()
        participation_counts = self.athletes_df.groupby(['NOC', 'Sport']).size()
        success_rate = (medal_counts / participation_counts).fillna(0)
        
        # 项目专长指数
        specialization = self.athletes_df[self.athletes_df['Medal'].notna()].groupby('NOC')['Sport'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        )
        
        sport_features['success_rate'] = success_rate
        sport_features['specialization'] = specialization
        return sport_features
    
    def _build_host_features(self):
        """构建主办国效应特征"""
        host_features = pd.DataFrame()
        
        # 合并主办国信息
        merged_data = pd.merge(self.medals_df, self.hosts_df, on='Year', how='left')
        
        # 计算主办国效应
        host_effect = merged_data[merged_data['NOC'] == merged_data['Host']].groupby('NOC').agg({
            'Total': lambda x: x.mean() / merged_data.groupby('NOC')['Total'].mean()
        }).rename(columns={'Total': 'host_effect'})
        
        return host_effect
    
    def _build_coach_features(self):
        """构建教练效应特征"""
        # 注意：这里需要根据实际数据情况调整
        # 这里示例使用运动员成绩变化作为替代指标
        coach_features = pd.DataFrame()
        
        # 计算运动项目的年度进步率
        performance_change = self.athletes_df.groupby(['NOC', 'Sport', 'Year'])['Medal'].apply(
            lambda x: (x != 'No Medal').mean()
        ).unstack(level=-1).pct_change(axis=1).mean(axis=1)
        
        coach_features['performance_improvement'] = performance_change
        return coach_features