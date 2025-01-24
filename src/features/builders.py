# src/features/builders.py
import pandas as pd
import numpy as np
from src.utils.logger import log_info, log_success

class FeatureBuilder:
    def __init__(self, medals_df, athletes_df, hosts_df, programs_df):
        self.medals_df = medals_df
        self.athletes_df = athletes_df
        self.hosts_df = hosts_df
        self.programs_df = programs_df

    def build_country_features(self):
        """构建国家特征"""
        log_info("构建国家历史表现特征")
        
        features = {}
        
        # 历史表现
        country_stats = self.medals_df.groupby('NOC').agg({
            'Gold': ['mean', 'std', 'max', 'sum'],
            'Total': ['mean', 'std', 'max', 'sum'],
            'Year': 'count'
        }).round(2)
        
        # 扁平化列名
        country_stats.columns = [
            f'{col[0]}_{col[1]}' for col in country_stats.columns
        ]
        
        features['country_stats'] = country_stats
        
        # 近期表现
        recent_years = self.medals_df[
            self.medals_df['Year'] >= 2000
        ].groupby('NOC').agg({
            'Gold': 'mean',
            'Total': 'mean'
        }).round(2)
        
        features['recent_performance'] = recent_years
        
        log_success("国家特征构建完成")
        return features

    def build_host_effect_features(self):
        """构建主办国效应特征"""
        log_info("构建主办国效应特征")
        
        # 合并主办国信息
        df = self.medals_df.merge(
            self.hosts_df,
            on='Year',
            how='left'
        )
        
        # 计算主办国效应
        host_effect = pd.DataFrame()
        host_effect['host_boost'] = df[df['NOC'] == df['Host']].groupby('NOC')['Total'].mean() / \
                                  df.groupby('NOC')['Total'].mean()
        
        log_success("主办国效应特征构建完成")
        return host_effect

    def build_sport_features(self):
        """构建运动项目特征"""
        log_info("构建运动项目特征")
        
        # 统计各国在不同项目上的优势
        sport_strength = self.athletes_df[
            self.athletes_df['Medal'].notna()
        ].groupby(['NOC', 'Sport'])['Medal'].count().unstack(fill_value=0)
        
        # 计算项目多样性
        sport_diversity = sport_strength.apply(
            lambda x: len(x[x > 0]), axis=1
        )
        
        log_success("运动项目特征构建完成")
        return {
            'sport_strength': sport_strength,
            'sport_diversity': sport_diversity
        }

    def build_time_series_features(self):
        """构建时间序列特征"""
        log_info("构建时间序列特征")
        
        # 按国家和年份聚合
        ts_features = self.medals_df.pivot_table(
            index='Year',
            columns='NOC',
            values=['Gold', 'Total'],
            aggfunc='sum',
            fill_value=0
        )
        
        # 处理移动平均
        rolling_mean = ts_features.rolling(window=3, min_periods=1).mean()
        
        # 优化趋势计算，避免逐列插入
        trend_dfs = []
        for col in ts_features.columns:
            series = ts_features[col]
            trend_df = pd.DataFrame(
                series.pct_change().fillna(0),
                columns=[col]
            )
            trend_dfs.append(trend_df)
        
        # 使用 concat 一次性合并所有趋势数据
        trends = pd.concat(trend_dfs, axis=1)
        
        # 填充无限值
        rolling_mean = rolling_mean.replace([np.inf, -np.inf], np.nan).fillna(0)
        trends = trends.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        log_success("时间序列特征构建完成")
        return {
            'rolling_mean': rolling_mean,
            'trends': trends
        }

    def build_all_features(self):
        """构建所有特征"""
        log_info("开始构建所有特征")
        
        features = {
            'country': self.build_country_features(),
            'host': self.build_host_effect_features(),
            'sport': self.build_sport_features(),
            'time_series': self.build_time_series_features()
        }
        
        log_success("所有特征构建完成")
        return features