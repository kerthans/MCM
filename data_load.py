from pathlib import Path
from rich.console import Console
from rich.table import Table
from src.data.data_loader import OlympicDataLoader
from src.data.preprocessor import OlympicDataPreprocessor
import pandas as pd

def main():
    console = Console(force_terminal=True)
    
    # 初始化数据加载器和预处理器
    console.print("\n[bold cyan]开始奥运数据处理流程[/bold cyan]")
    
    # 创建数据目录
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    console.print("\n[bold green]第一步：加载数据[/bold green]")
    loader = OlympicDataLoader()
    raw_data = loader.load_all_data()
    
    # 显示数据信息
    loader.display_data_info()
    
    # 显示内存使用情况
    console.print("\n[bold green]内存使用情况:[/bold green]")
    console.print(loader.get_memory_usage())
    
    # 初始化预处理器
    preprocessor = OlympicDataPreprocessor()
    
    # 预处理每个数据集
    console.print("\n[bold green]第二步：预处理各个数据集[/bold green]")
    processed_data = {
        'athletes': preprocessor.preprocess_athletes(raw_data['athletes']),
        'medal_counts': preprocessor.preprocess_medal_counts(raw_data['medal_counts']),
        'hosts': preprocessor.preprocess_hosts(raw_data['hosts']),
        'programs': preprocessor.preprocess_programs(raw_data['programs'])
    }
    
    # 创建组合特征
    console.print("\n[bold green]第三步：创建组合特征[/bold green]")
    combined_data = preprocessor.create_combined_features(processed_data)
    processed_data['combined'] = combined_data
    
    # 保存处理后的数据
    console.print("\n[bold green]第四步：保存处理后的数据[/bold green]")
    for name, df in processed_data.items():
        output_path = f"data/processed/{name}.parquet"
        df.to_parquet(output_path, index=False)
        console.print(f"已保存: {output_path}")
    
    # 显示处理摘要
    console.print("\n[bold cyan]处理摘要:[/bold cyan]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("数据集")
    summary_table.add_column("原始数据形状")
    summary_table.add_column("处理后数据形状")
    summary_table.add_column("新增特征")
    
    for name in raw_data.keys():
        raw_shape = raw_data[name].shape
        processed_shape = processed_data[name].shape
        new_features = set(processed_data[name].columns) - set(raw_data[name].columns)
        summary_table.add_row(
            name,
            f"{raw_shape[0]} × {raw_shape[1]}",
            f"{processed_shape[0]} × {processed_shape[1]}",
            ", ".join(new_features) if new_features else "无"
        )
    
    console.print(summary_table)
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()