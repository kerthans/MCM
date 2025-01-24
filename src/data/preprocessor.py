# src/data/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
from rich.progress import Progress, SpinnerColumn
from rich.panel import Panel
from rich.console import Console
from rich.tree import Tree
from datetime import datetime



class AdvancedDataPreprocessor:
    def __init__(self, console=None):
        self.console = console or Console()
        self.scalers = {}
        self.imputers = {}
        self.transformers = {}
        self.stats = {}
        
    def preprocess_all(self, data_dict):
        """高级数据预处理流程"""
        self.console.print(Panel("[bold cyan]开始高级数据预处理[/bold cyan]"))
        processed_data = {}
        
        with Progress(SpinnerColumn(), *Progress.get_default_columns()) as progress:
            task = progress.add_task("执行数据预处理...", total=len(data_dict))
            
            for name, df in data_dict.items():
                self.console.print(f"\n[yellow]处理数据集: {name}[/yellow]")
                # 创建深拷贝以避免修改原始数据
                df_copy = df.copy()
                
                # 保存原始年份数据
                if 'Year' in df_copy.columns:
                    year_data = df_copy['Year'].copy()
                
                # 进行其他预处理
                df_processed = self._preprocess_dataset(df_copy, name)
                
                # 还原年份数据
                if 'Year' in df_processed.columns:
                    df_processed['Year'] = year_data
                
                processed_data[name] = df_processed
                self._evaluate_preprocessing(df, df_processed, name)
                progress.update(task, advance=1)
                
        self._display_preprocessing_summary()
        return processed_data
    
    def _preprocess_dataset(self, df, name):
        """针对具体数据集的预处理"""
        df_processed = df.copy()
        
        # 分离需要标准化的数值列(排除Year列)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'Year' in numeric_cols:
            numeric_cols.remove('Year')
            
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # 处理数值列
        if numeric_cols:
            # 对非Year的数值列进行标准化
            self.scalers[f'{name}_robust'] = RobustScaler()
            df_processed[numeric_cols] = self.scalers[f'{name}_robust'].fit_transform(df_processed[numeric_cols])
        
        # 处理分类列
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0])
        
        return df_processed
    
    def _process_temporal_features(self, df):
        """处理时间相关特征"""
        if 'Year' in df.columns:
            # 添加周期性特征
            df['Olympic_Cycle'] = (df['Year'] - df['Year'].min()) // 4
            
            # 创建时代分类
            df['Era'] = pd.qcut(df['Year'], q=5, labels=['Very Early', 'Early', 'Middle', 'Recent', 'Modern'])
            
            # 相对时间特征
            df['Years_Since_First'] = df['Year'] - df['Year'].min()
            df['Years_To_Last'] = df['Year'].max() - df['Year']
            
        return df
    
    def _handle_missing_values(self, df, name):
        """高级缺失值处理"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            # 对数值列使用迭代式插补
            self.imputers[f'{name}_numeric'] = IterativeImputer(
                random_state=42,
                max_iter=10,
                skip_complete=True
            )
            df[numeric_cols] = self.imputers[f'{name}_numeric'].fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            # 对类别列使用众数填充
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        
        return df
    
    def _handle_outliers(self, df):
        """智能异常值处理"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 使用分位数进行温和的截断
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def _transform_features(self, df, name):
        """高级特征转换"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # 使用Yeo-Johnson变换处理偏态
            self.transformers[f'{name}_power'] = PowerTransformer(method='yeo-johnson')
            df[numeric_cols] = self.transformers[f'{name}_power'].fit_transform(df[numeric_cols])
            
            # 稳健缩放
            self.scalers[f'{name}_robust'] = RobustScaler()
            df[numeric_cols] = self.scalers[f'{name}_robust'].fit_transform(df[numeric_cols])
        
        return df
    
    def _evaluate_preprocessing(self, original_df, processed_df, name):
        """评估预处理效果"""
        stats = {
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': processed_df.isnull().sum().sum(),
            'memory_usage_before': original_df.memory_usage(deep=True).sum() / 1024**2,
            'memory_usage_after': processed_df.memory_usage(deep=True).sum() / 1024**2
        }
        
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['skewness_before'] = original_df[numeric_cols].skew().mean()
            stats['skewness_after'] = processed_df[numeric_cols].skew().mean()
        
        self.stats[name] = stats
        
    def _display_preprocessing_summary(self):
        """显示预处理效果摘要"""
        summary_tree = Tree("[bold cyan]预处理效果摘要[/bold cyan]")
        
        for name, stats in self.stats.items():
            dataset_branch = summary_tree.add(f"[yellow]{name}[/yellow]")
            dataset_branch.add(f"缺失值处理: {stats['missing_values_before']} → {stats['missing_values_after']}")
            dataset_branch.add(f"内存占用(MB): {stats['memory_usage_before']:.2f} → {stats['memory_usage_after']:.2f}")
            if 'skewness_before' in stats:
                dataset_branch.add(f"偏度改善: {stats['skewness_before']:.2f} → {stats['skewness_after']:.2f}")
        
        self.console.print("\n")
        self.console.print(summary_tree)