import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

class OlympicsDataPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def preprocess_medal_counts(self, df):
        """预处理奖牌数据"""
        df_processed = df.copy()
        
        # 填充缺失值
        numeric_columns = ['Gold', 'Silver', 'Bronze', 'Total']
        df_processed[numeric_columns] = self.imputer.fit_transform(
            df_processed[numeric_columns]
        )
        
        # 添加主要特征
        df_processed['Gold_Ratio'] = df_processed['Gold'] / df_processed['Total']
        df_processed['Medal_Score'] = (df_processed['Gold'] * 3 + 
                                     df_processed['Silver'] * 2 + 
                                     df_processed['Bronze'])
        
        return df_processed

    def preprocess_athletes(self, df):
        """预处理运动员数据"""
        df_processed = df.copy()
        
        # 处理Medal列
        df_processed['Medal'] = df_processed['Medal'].fillna('No medal')
        
        # 创建参赛次数特征
        athlete_counts = df_processed.groupby(['Name', 'NOC'])['Year'].count()
        df_processed['Participations'] = df_processed.merge(
            athlete_counts, 
            on=['Name', 'NOC'], 
            how='left'
        )['Year_y']
        
        return df_processed

    def create_country_features(self, medals_df, athletes_df):
        """创建国家特征"""
        # 计算每个国家的历史表现
        country_stats = medals_df.groupby('NOC').agg({
            'Gold': ['mean', 'std', 'max'],
            'Total': ['mean', 'std', 'max'],
            'Year': 'count'
        }).round(2)
        
        # 计算运动项目多样性
        sport_diversity = athletes_df.groupby('NOC')['Sport'].nunique()
        
        return pd.concat([country_stats, sport_diversity], axis=1)