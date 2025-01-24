# main.py
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from src.data.loader import OlympicsDataLoader
from src.data.preprocessor import AdvancedDataPreprocessor
from src.features.advanced_builder import AdvancedFeatureBuilder
from src.models.base_models import OlympicModelBuilder
from src.models.model_trainer import ModelTrainer
from src.models.predictor import OlympicPredictor

def clean_index(index_series):
    """清理索引，移除无效值并转换为整数"""
    # 转换为数值类型
    numeric_index = pd.to_numeric(index_series, errors='coerce')
    # 移除 NA 和无限值
    valid_index = numeric_index[~numeric_index.isna() & ~np.isinf(numeric_index)]
    # 转换为整数
    return valid_index.astype(int)

def prepare_data(processed_data, features, console):
    """准备训练数据"""
    try:
        # 1. 处理年份数据
        console.print("[cyan]开始处理年份数据...[/cyan]")
        
        medal_data = processed_data['medal_counts'].copy()
        console.print(f"[cyan]原始奖牌数据形状: {medal_data.shape}[/cyan]")
        console.print(f"[cyan]奖牌数据列: {medal_data.columns.tolist()}[/cyan]")
        
        # 确保Year列为整数类型
        medal_data['Year'] = pd.to_numeric(medal_data['Year'], errors='coerce')
        years = medal_data['Year'].dropna().astype(int).unique()
        
        if not len(years):
            raise ValueError("没有找到年份数据")
            
        console.print(f"[cyan]有效年份范围: {min(years)} - {max(years)}[/cyan]")
        
        # 2. 准备时间序列数据
        console.print("[cyan]准备时间序列数据...[/cyan]")
        X_ts = pd.DataFrame(index=sorted(years))
        
        # 3. 处理奖牌数据
        console.print("[cyan]处理奖牌数据...[/cyan]")
        y_ts = medal_data.groupby('Year')['Total'].mean()
        console.print(f"[cyan]奖牌数据年份范围: {y_ts.index.min()} - {y_ts.index.max()}[/cyan]")
        console.print(f"[cyan]奖牌数据年份数量: {len(y_ts.index.unique())}[/cyan]")
        
        # 4. 处理特征数据
        console.print("[cyan]处理特征数据...[/cyan]")
        
        # 验证特征数据
        if not isinstance(features, dict):
            raise ValueError("特征数据格式错误，应为字典类型")
            
        # 选择合适的特征集
        feature_keys = features.keys()
        console.print(f"[cyan]可用特征集: {list(feature_keys)}[/cyan]")
        
        if 'time_series' in features:
            X_ml = features['time_series'].copy()
        elif 'statistical' in features:
            X_ml = features['statistical'].copy()
        else:
            # 使用第一个可用的特征集
            first_key = list(features.keys())[0]
            X_ml = features[first_key].copy()
            console.print(f"[yellow]使用 {first_key} 作为特征集[/yellow]")
        
        # 确保特征数据索引为年份
        if X_ml.index.dtype != 'int64' or (X_ml.index.min() < 1800 or X_ml.index.max() > 2100):
            console.print("[yellow]特征数据索引不是有效的年份，尝试重建索引...[/yellow]")
            # 使用与奖牌数据相同的年份重建索引
            valid_years = sorted(set(years))
            X_ml = X_ml.reset_index(drop=True)
            if len(X_ml) >= len(valid_years):
                X_ml.index = valid_years[:len(X_ml)]
            else:
                raise ValueError("特征数据行数小于有效年份数量")
        
        # 检查并打印特征数据信息
        console.print(f"[cyan]特征数据形状: {X_ml.shape}[/cyan]")
        console.print(f"[cyan]特征数据索引类型: {X_ml.index.dtype}[/cyan]")
        console.print(f"[cyan]特征数据年份范围: {X_ml.index.min()} - {X_ml.index.max()}[/cyan]")
        
        # 5. 数据对齐
        console.print("[cyan]对齐数据...[/cyan]")
        console.print(f"[cyan]y_ts索引: {sorted(y_ts.index.tolist())}[/cyan]")
        console.print(f"[cyan]X_ml索引: {sorted(X_ml.index.tolist())}[/cyan]")
        
        common_years = sorted(set(X_ml.index).intersection(set(y_ts.index)))
        console.print(f"[cyan]共同年份数量: {len(common_years)}[/cyan]")
        
        if not common_years:
            raise ValueError("没有找到共同的年份数据")
            
        if len(common_years) < 10:
            raise ValueError(f"有效数据量不足，当前仅有 {len(common_years)} 个样本")
        
        # 根据共同年份筛选数据
        X_ml = X_ml.loc[common_years]
        y_ml = y_ts.loc[common_years]
        
        # 6. 数据清理
        # 替换无穷大值
        X_ml = X_ml.replace([np.inf, -np.inf], np.nan)
        # 使用中位数填充缺失值
        X_ml = X_ml.fillna(X_ml.median())
        
        # 7. 打印最终数据信息
        console.print(f"[green]数据处理完成:[/green]")
        console.print(f"[cyan]- 最终数据范围: {min(common_years)} - {max(common_years)}[/cyan]")
        console.print(f"[cyan]- 有效样本数量: {len(common_years)}[/cyan]")
        console.print(f"[cyan]- 特征维度: {X_ml.shape}[/cyan]")
        console.print(f"[cyan]- 目标维度: {y_ml.shape}[/cyan]")
        
        return X_ts, y_ts, X_ml, y_ml
        
    except Exception as e:
        console.print(f"[red]数据准备失败: {str(e)}[/red]")
        console.print(f"[red]错误类型: {type(e).__name__}[/red]")
        console.print(f"[red]错误位置: {e.__traceback__.tb_frame.f_code.co_name}[/red]")
        raise

def main():
    # 添加警告过滤
    import warnings
    warnings.filterwarnings('ignore', category=Warning)
    
    console = Console()
    
    try:
        console.print(Panel("[bold cyan]奥运会数据分析系统[/bold cyan]", subtitle="初始化..."))
        
        # 1. 数据加载和预处理
        loader = OlympicsDataLoader()
        raw_data = loader.load_data("data/raw")
        
        # 打印原始数据信息
        console.print("[cyan]原始数据加载完成[/cyan]")
        for key, df in raw_data.items():
            console.print(f"[cyan]{key} 数据形状: {df.shape}[/cyan]")
            if 'Year' in df.columns:
                year_min = df['Year'].min() if not pd.isna(df['Year'].min()) else "N/A"
                year_max = df['Year'].max() if not pd.isna(df['Year'].max()) else "N/A"
                console.print(f"[cyan]{key} 年份范围: {year_min} - {year_max}[/cyan]")
        
        preprocessor = AdvancedDataPreprocessor(console)
        processed_data = preprocessor.preprocess_all(raw_data)
        
        # 打印预处理后的数据信息
        console.print("[cyan]数据预处理完成[/cyan]")
        for key, df in processed_data.items():
            if isinstance(df, pd.DataFrame):
                console.print(f"[cyan]{key} 数据形状: {df.shape}[/cyan]")
                if 'Year' in df.columns:
                    year_min = df['Year'].min() if not pd.isna(df['Year'].min()) else "N/A"
                    year_max = df['Year'].max() if not pd.isna(df['Year'].max()) else "N/A"
                    console.print(f"[cyan]{key} 年份范围: {year_min} - {year_max}[/cyan]")
        
        # 2. 特征工程
        feature_builder = AdvancedFeatureBuilder(
            processed_data['medal_counts'],
            processed_data['athletes'],
            processed_data['hosts'],
            processed_data['programs'],
            console
        )
        features = feature_builder.build_all_features()
        
        # 3. 准备训练数据
        X_ts, y_ts, X_ml, y_ml = prepare_data(processed_data, features, console)
        
        # 4. 模型构建和训练
        console.print("[cyan]开始模型构建和训练...[/cyan]")
        
        model_builder = OlympicModelBuilder(features, console)
        
        # 构建基础模型
        ts_models = model_builder.build_time_series_models()
        ml_models = model_builder.build_ml_models()
        
        trainer = ModelTrainer(
            {**ts_models, **ml_models},
            features,
            console
        )
        
        # 5. 模型训练
        try:
            # 训练时间序列模型
            ts_params, ts_scores = trainer.train_and_optimize(X_ts, y_ts, model_type='ts', max_time=300)  # 5分钟超时
            console.print("[green]✓ 时间序列模型训练完成[/green]")
            
            # 训练机器学习模型
            ml_params, ml_scores = trainer.train_and_optimize(X_ml, y_ml, model_type='ml', max_time=300)  # 5分钟超时
            console.print("[green]✓ 机器学习模型训练完成[/green]")
            
            # 条件性训练深度学习模型
            if len(X_ml) >= 30:
                dl_models = model_builder.build_deep_learning_models()
                trainer.models.update(dl_models)
                
                X_sequences, y_sequences = model_builder.prepare_sequence_data(
                    X_ml, 
                    y_ml,
                    min_samples=10
                )
                
                dl_params, dl_scores = trainer.train_and_optimize(
                    X_sequences, 
                    y_sequences, 
                    model_type='dl'
                )
                console.print("[green]✓ 深度学习模型训练完成[/green]")
            else:
                console.print("[yellow]数据量不足，跳过深度学习模型训练[/yellow]")
                dl_params, dl_scores = {}, {}
            
            # 6. 打印训练结果
            console.print("\n[cyan]模型训练结果摘要:[/cyan]")
            if ts_scores:
                console.print(f"时间序列模型得分: {ts_scores}")
            if ml_scores:
                console.print(f"机器学习模型得分: {ml_scores}")
            if dl_scores:
                console.print(f"深度学习模型得分: {dl_scores}")
            
        except Exception as e:
            console.print(f"[red]模型训练失败: {str(e)}[/red]")
            raise
        
    except Exception as e:
        console.print(Panel(f"[bold red]错误: {str(e)}[/bold red]", subtitle="❌ 处理失败"))
        raise

if __name__ == "__main__":
    main()