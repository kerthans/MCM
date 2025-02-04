import pandas as pd
import numpy as np
from pathlib import Path
from rich.progress import track
from rich.console import Console
from rich.table import Table
from typing import Dict, Optional

class OlympicDataLoader:
    def __init__(self, data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """初始化奥运数据加载器"""
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.console = Console(force_terminal=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, pd.DataFrame] = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有奥运数据集"""
        try:
            with self.console.status("[bold green]正在加载数据文件...") as status:
                # 加载运动员数据
                athletes_dtypes = {
                    'Name': 'string',
                    'Sex': 'category',
                    'Team': 'string',
                    'NOC': 'category',
                    'Year': 'int16',
                    'City': 'category',
                    'Sport': 'category',
                    'Event': 'string',
                    'Medal': 'category'
                }
                self.data['athletes'] = pd.read_csv(
                    self.data_dir / "summerOly_athletes.csv",
                    dtype=athletes_dtypes,
                    na_values=['NA', '']
                )
                self.console.log("运动员数据已加载")

                # 加载奖牌数据
                medal_dtypes = {
                    'NOC': 'string',
                    'Gold': 'int16',
                    'Silver': 'int16',
                    'Bronze': 'int16',
                    'Total': 'int16',
                    'Year': 'int16'
                }
                self.data['medal_counts'] = pd.read_csv(
                    self.data_dir / "summerOly_medal_counts.csv",
                    dtype=medal_dtypes
                )
                self.console.log("奖牌统计数据已加载")

                # 加载主办国数据
                self.data['hosts'] = pd.read_csv(
                    self.data_dir / "summerOly_hosts.csv",
                    encoding='utf-8'
                )
                self.data['hosts']['Year'] = self.data['hosts']['Year'].astype('int16')
                self.console.log("主办国数据已加载")

                # 加载项目数据 - 使用 encoding='latin1' 解决编码问题
                self.data['programs'] = pd.read_csv(
                    self.data_dir / "summerOly_programs.csv",
                    encoding='latin1'  # 修改这里的编码
                )
                self.console.log("项目数据已加载")

            return self.data
            
        except Exception as e:
            self.console.print(f"[bold red]数据加载错误: {str(e)}")
            raise e

    def display_data_info(self) -> None:
        """显示数据集信息"""
        for name, df in self.data.items():
            self.console.print(f"\n[bold cyan]{name} 数据集信息:[/bold cyan]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("字段名称")
            table.add_column("数据类型")
            table.add_column("非空值数量")
            table.add_column("唯一值数量")
            
            for col in df.columns:
                table.add_row(
                    col,
                    str(df[col].dtype),
                    f"{df[col].count()}/{len(df)}",
                    str(df[col].nunique())
                )
            
            self.console.print(table)

    def get_memory_usage(self) -> pd.DataFrame:
        """计算数据集内存使用情况"""
        memory_usage = {}
        for name, df in self.data.items():
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            memory_usage[name] = {
                '内存占用 (MB)': f"{memory_mb:.2f}",
                '行数': df.shape[0],
                '列数': df.shape[1]
            }
        return pd.DataFrame(memory_usage).T