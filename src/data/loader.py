# src/data/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from src.utils.logger import log_info, log_error, log_success

class OlympicsDataLoader:
    def __init__(self):
        self.console = Console(color_system="auto")
        self.data = {}
        self.data_info = {}
        self.encodings = {
            'athletes': 'utf-8',
            'medal_counts': 'utf-8',
            'hosts': 'utf-8',
            'programs': 'latin1',  # 修改编码
            'dictionary': 'latin1'  # 修改编码
        }
        
    def load_data(self, data_dir):
        """加载所有数据文件并进行完整性检查"""
        files = {
            'athletes': 'summerOly_athletes.csv',
            'medal_counts': 'summerOly_medal_counts.csv',
            'hosts': 'summerOly_hosts.csv',
            'programs': 'summerOly_programs.csv',
            'dictionary': 'data_dictionary.csv'
        }
        
        self.console.print(Panel("[bold cyan]开始数据加载流程[/bold cyan]", expand=False))
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            load_task = progress.add_task("[cyan]加载数据文件...", total=len(files))
            
            for key, filename in files.items():
                try:
                    file_path = Path(data_dir) / filename
                    self.console.print(f"[yellow]正在加载 {key} 数据集...")
                    
                    df = pd.read_csv(
                        file_path,
                        encoding=self.encodings[key],
                        on_bad_lines='warn'
                    )
                    
                    # 数据质量检查
                    df = self._check_data_quality(df, key)
                    
                    self.data[key] = df
                    self.data_info[key] = self._analyze_dataset(df, key)
                    
                    progress.update(load_task, advance=1)
                    log_success(f"成功加载 {key} 数据集: {len(df)} 行")
                    
                except Exception as e:
                    log_error(f"加载 {key} 数据集失败: {str(e)}")
                    raise
                    
        self._display_detailed_summary()
        return self.data
    
    def _check_data_quality(self, df, name):
        """详细的数据质量检查"""
        quality_tree = Tree(f"[bold cyan]{name} 数据质量检查[/bold cyan]")
        
        # 检查缺失值
        missing = df.isnull().sum()
        if missing.any():
            missing_branch = quality_tree.add("[yellow]存在缺失值的列[/yellow]")
            for col, count in missing[missing > 0].items():
                missing_branch.add(f"{col}: {count} 个缺失值 ({count/len(df)*100:.2f}%)")
        
        # 检查重复行
        duplicates = df.duplicated().sum()
        if duplicates:
            quality_tree.add(f"[yellow]发现 {duplicates} 行重复数据[/yellow]")
        
        # 检查异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outliers_branch = quality_tree.add("[yellow]异常值检测[/yellow]")
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].shape[0]
                if outliers:
                    outliers_branch.add(f"{col}: {outliers} 个潜在异常值")
        
        self.console.print(quality_tree)
        return df
    
    def _analyze_dataset(self, df, name):
        """分析数据集的详细信息"""
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': df.dtypes.value_counts().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        if 'Year' in df.columns:
            info['year_range'] = (df['Year'].min(), df['Year'].max())
            info['year_coverage'] = len(df['Year'].unique())
        
        return info
    
    def _display_detailed_summary(self):
        """显示详细的数据摘要信息"""
        self.console.print("\n[bold cyan]数据集加载摘要[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("数据集", style="cyan")
        table.add_column("行数", justify="right")
        table.add_column("列数", justify="right")
        table.add_column("缺失值", justify="right")
        table.add_column("内存占用(MB)", justify="right")
        table.add_column("数据类型", justify="left")
        
        for name, info in self.data_info.items():
            table.add_row(
                name,
                str(info['rows']),
                str(info['columns']),
                str(info['missing_values']),
                f"{info['memory_usage']:.2f}",
                str(list(info['dtypes'].keys()))
            )
            
        self.console.print(table)
        
        # 显示额外的统计信息
        for name, info in self.data_info.items():
            if 'year_range' in info:
                self.console.print(f"\n[cyan]{name}[/cyan] 时间跨度: "
                                 f"{info['year_range'][0]} - {info['year_range'][1]} "
                                 f"({info['year_coverage']} 个年份)")