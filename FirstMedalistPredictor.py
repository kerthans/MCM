import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FirstMedalistPredictor:
    def __init__(self):
        self.console = Console()
        # 配置基础模型
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def _prepare_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        features = []
        
        # 按国家分组处理
        for noc, country_data in historical_data.groupby('NOC'):
            country_data = country_data.sort_values('Year')
            
            # 基础特征
            latest_data = country_data.iloc[-1]
            feature_dict = {
                'NOC': noc,
                'Years_Participated': len(country_data),
                'Last_Participation': latest_data['Year'],
                'Best_Rank': country_data['Rank'].min(),
                'Average_Rank': country_data['Rank'].mean(),
                'Recent_Rank_Trend': country_data['Rank'].diff().tail(3).mean(),
                'GDP_Latest': latest_data.get('GDP', 0),
                'Population_Latest': latest_data.get('Population', 0),
                'Sports_Investment': latest_data.get('Sports_Investment', 0)
            }
            
            # 历史表现特征
            feature_dict.update({
                'Historical_Best_Total': country_data['Total'].max(),
                'Recent_Total_Avg': country_data['Total'].tail(3).mean(),
                'Total_Growth_Rate': (
                    country_data['Total'].diff().mean() / 
                    country_data['Total'].mean() if country_data['Total'].mean() > 0 else 0
                ),
                'Near_Medal_Times': sum(
                    (country_data['Total'] == 0) & 
                    (country_data['Rank'] <= 20)
                )
            })
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

    def identify_potential_countries(self, 
                                  historical_data: pd.DataFrame, 
                                  min_participations: int = 3) -> pd.DataFrame:
        """识别潜在的首次获奖国家"""
        # 获取历史获奖记录
        medalists = set(historical_data[historical_data['Total'] > 0]['NOC'].unique())
        
        # 筛选从未获奖的国家
        never_medaled = historical_data[~historical_data['NOC'].isin(medalists)]
        candidates = never_medaled.groupby('NOC').agg({
            'Year': ['count', 'max'],
            'Rank': ['min', 'mean']
        }).reset_index()
        
        candidates.columns = ['NOC', 'Participations', 'Last_Year', 'Best_Rank', 'Avg_Rank']
        
        # 筛选符合条件的国家
        qualified = candidates[
            (candidates['Participations'] >= min_participations) & 
            (candidates['Last_Year'] >= 2016)  # 确保近期仍在参赛
        ]
        
        print(f"\n[Debug] 潜在获奖国家筛选:")
        print(f"▪ 历史获奖国家数: {len(medalists)}")
        print(f"▪ 从未获奖国家数: {len(never_medaled['NOC'].unique())}")
        print(f"▪ 符合条件国家数: {len(qualified)}")
        
        return qualified

    def train_prediction_model(self, features_df: pd.DataFrame, historical_data: pd.DataFrame):
        """训练预测模型"""
        # 准备训练数据
        X = features_df.drop(['NOC'], axis=1)
        y = historical_data.groupby('NOC')['Total'].max()
        y = y[y > 0]  # 只用有奖牌的国家训练
        
        # 对齐数据
        common_nocs = set(features_df['NOC']) & set(y.index)
        X = X[features_df['NOC'].isin(common_nocs)]
        y = y[list(common_nocs)]
        
        print(f"\n[Debug] 模型训练信息:")
        print(f"▪ 训练样本数: {len(X)}")
        print(f"▪ 特征数量: {X.shape[1]-1}")  # 减去NOC列
        
        # 训练模型
        self.model.fit(X, y)
        
        # 特征重要性
        importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print("\n[Debug] 主要预测因素:")
        for feat, imp in importance.head().items():
            print(f"▪ {feat}: {imp:.3f}")
        
        return self.model

    def predict_medal_chances(self, 
                            potential_countries: pd.DataFrame,
                            features_df: pd.DataFrame) -> Dict[str, Dict]:
        """预测首次获奖可能性"""
        predictions = {}
        
        # 准备预测数据
        X_pred = features_df[features_df['NOC'].isin(potential_countries['NOC'])]
        feature_data = X_pred.drop(['NOC'], axis=1)
        
        # 预测
        raw_predictions = self.model.predict(feature_data)
        
        for idx, noc in enumerate(X_pred['NOC']):
            country_features = features_df[features_df['NOC'] == noc].iloc[0]
            pred_value = raw_predictions[idx]
            
            # 计算获奖概率
            base_prob = 1 / (1 + np.exp(-3 * (pred_value - 0.3)))
            experience_factor = min(country_features['Years_Participated'] / 10, 1.0)
            rank_factor = 1 / (1 + np.exp((country_features['Best_Rank'] - 15) / 5))
            
            final_probability = base_prob * experience_factor * rank_factor
            
            # 评估置信度
            if pred_value > 0.8 and country_features['Best_Rank'] < 12:
                confidence = 'high'
            elif pred_value > 0.5 and country_features['Best_Rank'] < 15:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            predictions[noc] = {
                'predicted_medals': pred_value,
                'probability': final_probability,
                'confidence': confidence,
                'years_participated': int(country_features['Years_Participated']),
                'best_rank': int(country_features['Best_Rank']),
                'recent_trend': 'improving' if country_features['Recent_Rank_Trend'] < 0 else 'stable'
            }
        
        return predictions

    def display_predictions(self, predictions: Dict[str, Dict]) -> None:
        """显示预测结果"""
        table = Table(
            title="\n2028年首次获奖国家预测",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("国家", justify="left")
        table.add_column("预测奖牌数", justify="center")
        table.add_column("获奖概率", justify="center")
        table.add_column("最佳排名", justify="center")
        table.add_column("参赛经验", justify="center")
        table.add_column("近期趋势", justify="right")
        table.add_column("置信度", justify="right")
        
        for country, info in sorted(
            predictions.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        ):
            confidence_color = {
                'high': '[green]',
                'medium': '[yellow]',
                'low': '[red]'
            }[info['confidence']]
            
            table.add_row(
                f"[bold]{country}[/bold]",
                f"{info['predicted_medals']:.1f}",
                f"{info['probability']:.1%}",
                str(info['best_rank']),
                f"{info['years_participated']}届",
                info['recent_trend'],
                f"{confidence_color}{info['confidence']}[/]"
            )
        
        self.console.print(table)


def main():
    console = Console()

    try:
        # 加载数据
        console.print("[bold cyan]加载历史数据...[/bold cyan]")

        # ========== 替换开始 ==========
        def try_multiple_extensions(base_path: str) -> Path:
            """尝试多种parquet扩展名"""
            extensions = ['', '.parquet', '.snappy.parquet', '.gzip.parquet']
            for ext in extensions:
                path = Path(base_path + ext)
                if path.exists():
                    return path
            raise FileNotFoundError(f"No parquet file found with base path: {base_path}")

        # 构建数据路径
        data_path = Path("data/processed/medal_counts")
        data_file = try_multiple_extensions(str(data_path))

        # 验证目录结构
        if not data_file.parent.exists():
            raise FileNotFoundError(
                f"数据目录不存在: {data_file.parent}\n"
                "请确认：\n"
                "1. 项目目录结构是否正确\n"
                "2. 是否已运行数据预处理流程\n"
                "3. 数据文件是否已下载到指定位置"
            )

        historical_data = pd.read_parquet(data_file)
        
        predictor = FirstMedalistPredictor()
        
        # 1. 识别潜在国家
        potential_countries = predictor.identify_potential_countries(historical_data)
        
        # 2. 准备特征
        features_df = predictor._prepare_features(historical_data)
        
        # 3. 训练模型
        predictor.train_prediction_model(features_df, historical_data)
        
        # 4. 预测获奖概率
        predictions = predictor.predict_medal_chances(potential_countries, features_df)
        
        # 5. 显示结果
        predictor.display_predictions(predictions)
        
        # 6. 保存结果
        predictions_df = pd.DataFrame.from_dict(predictions, orient='index')
        predictions_df.to_csv('models/first_time_medalists_2028.csv')
        
        console.print("\n[bold green]预测完成! 结果已保存到 models/first_time_medalists_2028.csv[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()