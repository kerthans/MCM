from pathlib import Path
from rich.console import Console
import pandas as pd
from src.features.feature_engineer import OlympicFeatureEngineer

def main():
    console = Console()
    
    try:
        # 加载预处理后的数据
        console.print("\n[bold cyan]加载预处理数据...[/bold cyan]")
        
        # 尝试读取parquet文件，如果失败则读取csv
        try:
            medal_data = pd.read_parquet("data/processed/medal_counts.parquet")
            host_data = pd.read_parquet("data/processed/hosts.parquet")
            program_data = pd.read_parquet("data/processed/programs.parquet")
            console.print("[green]成功从parquet文件加载数据[/green]")
        except:
            console.print("[yellow]parquet文件读取失败，尝试读取CSV文件[/yellow]")
            medal_data = pd.read_csv("data/processed/medal_counts.csv")
            host_data = pd.read_csv("data/processed/hosts.csv")
            program_data = pd.read_csv("data/processed/programs.csv")
            console.print("[green]成功从CSV文件加载数据[/green]")
        
        # 显示数据基本信息
        console.print("\n[bold cyan]数据加载完成，基本信息：[/bold cyan]")
        console.print(f"奖牌数据形状: {medal_data.shape}")
        console.print(f"主办国数据形状: {host_data.shape}")
        console.print(f"项目数据形状: {program_data.shape}")
        
        # 初始化特征工程器
        engineer = OlympicFeatureEngineer()
        
        # 构建特征
        features = engineer.engineer_all_features(
            medal_data=medal_data,
            host_data=host_data,
            program_data=program_data
        )
        
        # 保存特征
        output_path = "data/processed/features.parquet"
        features.to_csv(output_path, index=False, encoding='utf-8')
        console.print(f"\n[bold green]特征已保存至：{output_path}[/bold green]")
        
        # 显示特征信息
        console.print("\n[bold cyan]特征工程完成，特征概览：[/bold cyan]")
        console.print(f"特征数量: {features.shape[1]}")
        console.print(f"样本数量: {features.shape[0]}")
        
        return features
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        raise e

if __name__ == "__main__":
    features = main()