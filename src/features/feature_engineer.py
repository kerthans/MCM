import warnings
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import track
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

class OlympicFeatureEngineer:
    def __init__(self):
        self.console = Console()
        self.scaler = StandardScaler()
        
    def create_country_strength_features(self, 
                                       medal_data: pd.DataFrame, 
                                       window_sizes: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """创建国家实力相关特征"""
        df = medal_data.copy()
        
        with self.console.status("[bold green]构建国家实力特征...") as status:
            # 对每个国家计算移动平均和趋势
            for window in window_sizes:
                # 计算过去几届的平均金牌数
                df[f'Gold_MA_{window}'] = df.groupby('NOC')['Gold'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
                
                # 计算过去几届的平均奖牌总数
                df[f'Total_MA_{window}'] = df.groupby('NOC')['Total'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
                
                # 计算变化率
                df[f'Gold_Change_{window}'] = df.groupby('NOC')['Gold'].transform(
                    lambda x: x.pct_change(periods=window)
                )
                
            # 计算历史最好成绩
            df['Historical_Best_Gold'] = df.groupby('NOC')['Gold'].transform(
                lambda x: x.expanding().max().shift(1)
            )
            
            # 计算参与连续性
            df['Olympics_Since_Last'] = df.groupby('NOC')['Year'].transform(
                lambda x: x.diff()
            )
            
            self.console.log("国家实力特征构建完成")
            return df
            
    def create_host_effect_features(self, 
                                  medal_data: pd.DataFrame, 
                                  host_data: pd.DataFrame) -> pd.DataFrame:
        """创建主办国效应特征"""
        df = medal_data.copy()
        
        with self.console.status("[bold green]构建主办国效应特征...") as status:
            # 合并主办国信息
            hosts = host_data[['Year', 'Country']].copy()
            df = df.merge(hosts, on='Year', how='left')
            
            # 创建主办国标志
            df['Is_Host'] = (df['NOC'] == df['Country']).astype(int)
            
            # 计算主办前后的表现变化
            df['Host_Year'] = df.groupby('NOC')['Is_Host'].transform(
                lambda x: x.rolling(3, center=True).sum()
            )
            
            # 计算地理位置效应（同一洲）
            # 这里需要额外的国家-洲际映射数据，暂时略过
            
            self.console.log("主办国效应特征构建完成")
            return df
            
    # def create_sport_structure_features(self,
    #                                   medal_data: pd.DataFrame,
    #                                   program_data: pd.DataFrame) -> pd.DataFrame:
    #     """创建项目结构特征"""
    #     df = medal_data.copy()
    #
    #     with self.console.status("[bold green]构建项目结构特征...") as status:
    #         # 计算每年的项目总数
    #         yearly_events = program_data[program_data.columns[
    #             program_data.columns.str.match(r'^\d{4}$')]].sum()
    #
    #         # 合并项目数信息
    #         event_counts = pd.DataFrame({
    #             'Year': yearly_events.index.astype(int),
    #             'Total_Events': yearly_events.values
    #         })
    #         df = df.merge(event_counts, on='Year', how='left')
    #
    #         # 计算奖牌效率
    #         df['Medal_Efficiency'] = df['Total'] / df['Total_Events']
    #         df['Gold_Efficiency'] = df['Gold'] / df['Total_Events']
    #
    #         self.console.log("项目结构特征构建完成")
    #         return df

    def create_sport_structure_features(self,
                                        medal_data: pd.DataFrame,
                                        program_data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced sport structure feature engineering"""
        df = medal_data.copy()

        with self.console.status("[bold green]构建项目结构特征...") as status:
            # 基础项目数量特征
            yearly_events = program_data[program_data.columns[
                program_data.columns.str.match(r'^\d{4}$')]].sum()

            event_counts = pd.DataFrame({
                'Year': yearly_events.index.astype(int),
                'Total_Events': yearly_events.values
            })
            df = df.merge(event_counts, on='Year', how='left')

            # 计算项目变化率
            df['Events_Change'] = df.groupby('NOC')['Total_Events'].pct_change()

            # 计算奖牌效率指标
            df['Medal_Efficiency'] = df['Total'] / df['Total_Events']
            df['Gold_Efficiency'] = df['Gold'] / df['Total_Events']

            # 计算项目多样性指标
            program_diversity = program_data.nunique(axis=1)
            df['Sport_Diversity'] = df['Year'].map(
                dict(zip(program_data.index, program_diversity))
            )

            # 添加项目结构相关性
            sport_correlations = program_data.corr()
            df['Sport_Correlation'] = df['Year'].map(
                dict(zip(program_data.index, sport_correlations.mean()))
            )

            # 计算优势项目占比
            def calculate_strength_ratio(row):
                if row['Total'] == 0:
                    return 0
                return row['Gold'] / row['Total']

            df['Strength_Ratio'] = df.apply(calculate_strength_ratio, axis=1)

            self.console.log("项目结构特征构建完成")
            return df
    def create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间序列特征"""
        with self.console.status("[bold green]构建时间序列特征...") as status:
            # 创建周期特征
            df['Years_To_Next'] = df.groupby('NOC')['Year'].transform(
                lambda x: x.shift(-1) - x
            )
            
            # 计算累计参与次数
            df['Participation_Count'] = df.groupby('NOC').cumcount() + 1
            
            # 计算上届排名
            df['Previous_Rank'] = df.groupby('Year')['Gold'].transform(
                lambda x: x.rank(ascending=False)
            ).shift(1)
            
            self.console.log("时间序列特征构建完成")
            return df
    
    def select_and_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征选择和处理"""
        with self.console.status("[bold green]特征处理与选择...") as status:
            # 选择最终使用的特征
            feature_cols = [
                # 基础特征
                'Year', 'NOC',
                
                # 国家实力特征
                'Gold_MA_1', 'Gold_MA_2', 'Gold_MA_3',
                'Total_MA_1', 'Total_MA_2', 'Total_MA_3',
                'Gold_Change_1', 'Gold_Change_2', 'Gold_Change_3',
                'Historical_Best_Gold', 'Olympics_Since_Last',
                
                # 主办国效应特征
                'Is_Host', 'Host_Year',
                
                # 项目结构特征
                'Medal_Efficiency', 'Gold_Efficiency', 'Total_Events',
                
                # 时间序列特征
                'Years_To_Next', 'Participation_Count', 'Previous_Rank'
            ]
            
            # 选择特征
            features_df = df[feature_cols].copy()
            
            # 处理缺失值和异常值
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            # 替换无穷大和极端值
            numeric_features = features_df.select_dtypes(include=[np.number]).columns
            for col in numeric_features:
                if col != 'Year':
                    # 处理无穷大和极端值
                    features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # 使用安全的填充方法
                    median_val = features_df[col].median()
                    features_df[col] = features_df[col].fillna(median_val)
                    
                    # 处理极端异常值（可选）
                    Q1 = features_df[col].quantile(0.25)
                    Q3 = features_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    features_df.loc[features_df[col] < lower_bound, col] = lower_bound
                    features_df.loc[features_df[col] > upper_bound, col] = upper_bound
            
            # 标准化数值特征（使用更健壮的缩放器）
            numeric_features = numeric_features.drop(['Year'])  # 不标准化年份
            
            features_df[numeric_features] = self.scaler.fit_transform(
                features_df[numeric_features]
            )
            
            self.console.log("特征处理与选择完成")
            return features_df

    def engineer_all_features(self,
                            medal_data: pd.DataFrame,
                            host_data: pd.DataFrame,
                            program_data: pd.DataFrame) -> pd.DataFrame:
        """构建所有特征"""
        self.console.print("\n[bold cyan]开始特征工程流程[/bold cyan]")
        
        # 1. 构建国家实力特征
        df = self.create_country_strength_features(medal_data)
        
        # 2. 构建主办国效应特征
        df = self.create_host_effect_features(df, host_data)
        
        # 3. 构建项目结构特征
        df = self.create_sport_structure_features(df, program_data)
        
        # 4. 构建时间序列特征
        df = self.create_time_series_features(df)
        
        # 5. 特征选择和处理
        final_features = self.select_and_process_features(df)
        
        # 构建特征说明
        feature_description = self.get_feature_description(final_features)
        self.console.print("\n[bold green]特征构建完成，特征说明：[/bold green]")
        self.console.print(feature_description)
        
        return final_features
    
    def get_feature_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成特征说明"""
        descriptions = {
            'Year': '奥运会年份',
            'NOC': '国家代码',
            'Gold_MA_1': '上一届金牌数',
            'Gold_MA_2': '前两届金牌平均数',
            'Gold_MA_3': '前三届金牌平均数',
            'Total_MA_1': '上一届奖牌总数',
            'Total_MA_2': '前两届奖牌总数平均数',
            'Total_MA_3': '前三届奖牌总数平均数',
            'Gold_Change_1': '金牌数相比上届变化率',
            'Gold_Change_2': '金牌数相比前两届变化率',
            'Gold_Change_3': '金牌数相比前三届变化率',
            'Historical_Best_Gold': '历史最佳金牌数',
            'Olympics_Since_Last': '距离上次参赛间隔',
            'Is_Host': '是否主办国',
            'Host_Year': '主办年份标记',
            'Medal_Efficiency': '奖牌效率（总数/项目数）',
            'Gold_Efficiency': '金牌效率',
            'Total_Events': '项目总数',
            'Years_To_Next': '距下届时间',
            'Participation_Count': '参与次数',
            'Previous_Rank': '上届排名'
        }
        
        return pd.DataFrame({
            '特征名': descriptions.keys(),
            '说明': descriptions.values()
        })