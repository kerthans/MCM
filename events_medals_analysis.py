import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import joblib

from src.visualization.EventsAnalysisVisualizer import EventsAnalysisVisualizer


class EventsMedalAnalyzer:
    def __init__(self):
        self.console = Console()
        self.sports_importance = {}
        self.events_impact = {}
        self.visualizer = EventsAnalysisVisualizer()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载必要的数据文件"""
        try:
            def try_load_data(file_path_base):
                """尝试多种方式加载数据"""
                attempts = [
                    (f"{file_path_base}.parquet", lambda x: pd.read_parquet(x)),
                    (f"{file_path_base}.csv", lambda x: pd.read_csv(x)),
                    (f"{file_path_base}.csv", lambda x: pd.read_csv(x, encoding='utf-8')),
                    (f"{file_path_base}.csv", lambda x: pd.read_csv(x, encoding='latin1'))
                ]
                
                last_error = None
                for file_path, reader in attempts:
                    try:
                        if Path(file_path).exists():
                            data = reader(file_path)
                            self.console.print(f"[green]成功从 {file_path} 加载数据[/green]")
                            return data
                    except Exception as e:
                        last_error = e
                        continue
                
                available_files = list(Path(file_path_base).parent.glob("*"))
                error_msg = (
                    f"无法加载数据文件 {file_path_base}.*\n"
                    f"当前目录下的文件: {[f.name for f in available_files] if available_files else '无'}"
                )
                raise FileNotFoundError(error_msg)

            # 加载数据
            medals_df = try_load_data("data/processed/medal_counts")
            events_raw_df = try_load_data("data/processed/programs")
            athletes_df = try_load_data("data/processed/athletes")
            
            # 数据验证
            def validate_dataframe(df: pd.DataFrame, name: str, required_cols: List[str]):
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    self.console.print(f"[yellow]警告: {name} 的列: {', '.join(df.columns)}[/yellow]")
                    raise ValueError(f"{name} 缺少必要的列: {', '.join(missing_cols)}")
            
            # 处理 events_df 的年份数据
            year_columns = [col for col in events_raw_df.columns if str(col).isdigit() or 
                        (isinstance(col, str) and col.replace('*','').isdigit())]
            
            # 重塑数据为长格式
            events_df = events_raw_df.melt(
                id_vars=['Sport', 'Discipline', 'Code', 'Sports Governing Body'],
                value_vars=year_columns,
                var_name='Year',
                value_name='Events'
            )
            
            # 清理年份数据
            events_df['Year'] = events_df['Year'].str.replace('*', '').astype(int)
            
            # 聚合每年的总项目数
            events_df = events_df.groupby('Year')['Events'].sum().reset_index()
            events_df = events_df.rename(columns={'Events': 'Total'})
            
            # 验证数据
            try:
                validate_dataframe(medals_df, "medals_df", ['Year', 'NOC', 'Gold', 'Total'])
                
                # 调试信息
                self.console.print(f"\n[cyan]数据集列信息:[/cyan]")
                self.console.print(f"medal_counts 列: {', '.join(medals_df.columns)}")
                self.console.print(f"programs 列: {', '.join(events_df.columns)}")
                self.console.print(f"athletes 列: {', '.join(athletes_df.columns)}")
                
                return medals_df, events_df, athletes_df
                
            except Exception as e:
                self.console.print(f"[bold red]数据验证错误: {str(e)}[/bold red]")
                raise
                
        except Exception as e:
            self.console.print(f"[bold red]数据加载错误: {str(e)}[/bold red]")
            raise

    def analyze_events_medals_correlation(self, medals_df: pd.DataFrame, events_df: pd.DataFrame) -> Dict:
        """分析赛事数量与奖牌数的相关关系"""
        results = {}

        merged_df = pd.merge(medals_df, events_df[['Year', 'Total']].rename(columns={'Total': 'Total_Events'}),
                             on='Year')

        # 计算整体相关系数
        yearly_totals = merged_df.groupby('Year').agg({
            'Total': 'sum',
            'Total_Events': 'first'
        })
        results['overall_correlation'] = yearly_totals['Total'].corr(yearly_totals['Total_Events'])

        # 选择主要国家
        top_countries = merged_df.groupby('NOC')['Total'].sum().nlargest(20).index

        country_correlations = {}
        for country in top_countries:
            country_data = merged_df[merged_df['NOC'] == country]
            if len(country_data) > 5:
                corr = country_data['Total'].corr(country_data['Total_Events'])
                if not pd.isna(corr):
                    country_correlations[country] = corr

        results['country_correlations'] = dict(sorted(
            country_correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10])

        return results

    def identify_key_sports(self, athletes_df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[str]]:
        """识别各国的重点运动项目"""
        # 计算总奖牌数以选择主要国家
        total_medals = athletes_df[athletes_df['Medal'].notna()].groupby('NOC', observed=True).size()
        major_countries = total_medals.nlargest(30).index

        # 计算各国各项目奖牌
        medals_by_sport = athletes_df[
            (athletes_df['Medal'].notna()) &
            (athletes_df['NOC'].isin(major_countries))
            ].groupby(['NOC', 'Sport'], observed=True).size().reset_index(name='medal_count')

        # 计算全球排名
        global_rankings = {}
        for sport in athletes_df['Sport'].unique():
            sport_medals = athletes_df[
                (athletes_df['Sport'] == sport) &
                (athletes_df['Medal'].notna())
                ].groupby('NOC', observed=True).size().sort_values(ascending=False)
            global_rankings[sport] = {noc: rank + 1 for rank, noc in enumerate(sport_medals.index)}

        sports_strength = {}
        for noc in major_countries:
            country_data = medals_by_sport[medals_by_sport['NOC'] == noc]
            total_medals = country_data['medal_count'].sum()

            if total_medals > 0:
                sports_scores = []
                for _, row in country_data.iterrows():
                    sport = row['Sport']
                    medal_count = row['medal_count']

                    sports_scores.append({
                        'sport': sport,
                        'medal_count': medal_count,
                        'percentage': medal_count / total_medals * 100,
                        'global_rank': global_rankings.get(sport, {}).get(noc, len(global_rankings.get(sport, {})) + 1)
                    })

                sports_scores.sort(key=lambda x: (x['medal_count'], x['percentage'], -x['global_rank']), reverse=True)
                sports_strength[noc] = sports_scores[:top_n]

        return sports_strength

    def _calculate_global_rank(self, athletes_df: pd.DataFrame, noc: str, sport: str) -> int:
        """计算某个国家在特定运动项目上的全球排名"""
        # 使用observed=True解决FutureWarning
        sport_rankings = athletes_df[
            (athletes_df['Sport'] == sport) &
            (athletes_df['Medal'].notna())
            ].groupby('NOC', observed=True).size().sort_values(ascending=False)

        return list(sport_rankings.index).index(noc) + 1 if noc in sport_rankings.index else len(sport_rankings) + 1

    def analyze_host_country_impact(self, medals_df: pd.DataFrame, events_df: pd.DataFrame,
                                    hosts_df: pd.DataFrame) -> Dict:
        """分析主办国对赛事设置的影响"""
        results = {'host_effects': pd.DataFrame()}

        try:
            # 数据准备和验证
            required_columns = {
                'medals_df': ['Year', 'NOC', 'Total'],
                'hosts_df': ['Year', 'Host']
            }

            for df_name, cols in required_columns.items():
                df = locals()[df_name]
                missing = [col for col in cols if col not in df.columns]
                if missing:
                    raise ValueError(f"{df_name}缺少列: {missing}")

            # 标准化国家代码
            hosts_df = hosts_df.copy()
            hosts_df['Host'] = hosts_df['Host'].str.upper()
            medals_df = medals_df.copy()
            medals_df['NOC'] = medals_df['NOC'].str.upper()

            # 合并数据
            merged_df = pd.merge(
                medals_df,
                hosts_df[['Year', 'Host']],
                on='Year',
                how='inner'
            )

            host_effects = []
            for year in merged_df['Year'].unique():
                host = merged_df[merged_df['Year'] == year]['Host'].iloc[0]

                # 获取前后12年的数据
                year_range = range(year - 12, year + 13, 4)
                host_data = medals_df[
                    (medals_df['NOC'] == host) &
                    (medals_df['Year'].isin(year_range))
                    ].sort_values('Year')

                if len(host_data) >= 3:
                    hosting_year = host_data[host_data['Year'] == year]
                    before_hosting = host_data[host_data['Year'] < year]['Total'].mean()
                    after_hosting = host_data[host_data['Year'] > year]['Total'].mean()

                    if not hosting_year.empty and not pd.isna(before_hosting):
                        effect = {
                            'Year': year,
                            'Host': host,
                            'Before': round(before_hosting, 2),
                            'During': hosting_year['Total'].iloc[0],
                            'After': round(after_hosting, 2) if not pd.isna(after_hosting) else None,
                            'Impact': round(hosting_year['Total'].iloc[0] - before_hosting, 2)
                        }
                        host_effects.append(effect)

            if host_effects:
                results['host_effects'] = pd.DataFrame(host_effects).sort_values('Year', ascending=False)
                results['host_effects'] = results['host_effects'].round(2)

        except Exception as e:
            self.console.print(f"[yellow]警告: 主办国影响分析出现问题: {str(e)}[/yellow]")

        return results

    def generate_report(self, correlation_results: Dict, sports_strength: Dict, host_impact: Dict) -> str:
        """生成分析报告"""
        report = []
        
        try:
            # 1. 总体关系分析
            report.append("1. 赛事数量与奖牌数关系分析")
            report.append(f"整体相关系数: {correlation_results.get('overall_correlation', 'N/A'):.3f}")
            report.append("\n各国相关性:")
            for country, corr in correlation_results.get('country_correlations', {}).items():
                report.append(f"  - {country}: {corr:.3f}")
            
            # 2. 重点运动项目分析
            report.append("\n2. 各国重点运动项目分析")
            for country, sports in list(sports_strength.items())[:10]:
                report.append(f"\n{country}的优势项目:")
                for sport in sports:
                    report.append(
                        f"  - {sport['sport']}: {sport['medal_count']}枚奖牌 "
                        f"({sport['percentage']:.1f}%), 全球第{sport['global_rank']}名"
                    )
            
            # 3. 主办国影响分析
            report.append("\n3. 主办国影响分析")
            host_effects = host_impact.get('host_effects', pd.DataFrame())
            
            if not host_effects.empty and 'Impact' in host_effects.columns:
                avg_impact = host_effects['Impact'].mean()
                report.append(f"平均主办国效应: {avg_impact:.2f}枚奖牌")
                
                recent_hosts = host_effects.nlargest(5, 'Impact')
                report.append("\n最显著的主办国效应:")
                for _, host in recent_hosts.iterrows():
                    report.append(
                        f"  - {host['Year']} {host['Host']}: "
                        f"增加{host['Impact']:.1f}枚奖牌"
                    )
            else:
                report.append("主办国数据不足，无法分析影响")
        
        except Exception as e:
            report.append(f"\n警告: 报告生成过程中出现错误: {str(e)}")
        
        return "\n".join(report)

def main():
    console = Console()
    
    try:
        # 初始化分析器
        analyzer = EventsMedalAnalyzer()
        
        # 加载数据
        console.print("[bold cyan]加载数据...[/bold cyan]")
        medals_df, events_df, athletes_df = analyzer.load_data()
        
        # 加载主办国数据
        def try_load_data(file_path_base):
            """尝试多种方式加载数据"""
            # 修正相对路径
            data_dir = Path("data/processed")
            file_path_base = data_dir / Path(file_path_base).name
            
            attempts = [
                (f"{file_path_base}.parquet", lambda x: pd.read_parquet(x)),
                (f"{file_path_base}.csv", lambda x: pd.read_csv(x))
            ]
            
            for file_path, reader in attempts:
                if Path(file_path).exists():
                    return reader(file_path)
            
            # 检查目录内容
            if data_dir.exists():
                available_files = list(data_dir.glob("*"))
                files_str = "\n".join(str(f.relative_to(data_dir)) for f in available_files)
                raise FileNotFoundError(
                    f"无法找到数据文件: {file_path_base}.*\n"
                    f"data/processed/ 目录下的文件:\n{files_str}"
                )
            else:
                raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        try:
            hosts_df = try_load_data("hosts")
        except FileNotFoundError as e:
            # 如果找不到hosts文件，创建一个基本的hosts数据框
            console.print("[yellow]警告: 无法加载hosts数据，使用基本数据替代[/yellow]")
            hosts_df = pd.DataFrame({
                'Year': medals_df['Year'].unique(),
                'Host': ['Unknown'] * len(medals_df['Year'].unique())
            })
        
        # 进行分析
        console.print("[bold cyan]分析赛事与奖牌关系...[/bold cyan]")
        correlation_results = analyzer.analyze_events_medals_correlation(medals_df, events_df)
        analyzer.visualizer.plot_events_correlation(correlation_results)
        console.print("[bold cyan]识别各国重点运动项目...[/bold cyan]")
        sports_strength = analyzer.identify_key_sports(athletes_df)
        analyzer.visualizer.plot_key_sports_analysis(sports_strength)
        console.print("[bold cyan]分析主办国影响...[/bold cyan]")
        host_impact = analyzer.analyze_host_country_impact(medals_df, events_df, hosts_df)
        analyzer.visualizer.plot_host_impact(host_impact)
        # 生成报告
        report = analyzer.generate_report(correlation_results, sports_strength, host_impact)
        
        # 保存报告
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "events_medals_analysis_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
        
        # 显示报告
        console.print("\n[bold green]分析报告:[/bold green]")
        console.print(report)
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        raise e
if __name__ == "__main__":
    main()