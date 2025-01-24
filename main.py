# main.py
from rich.traceback import install
from src.data.loader import OlympicsDataLoader
from src.features.builders import FeatureBuilder
from src.models.predictor import OlympicsPredictor
from src.utils.logger import log_info, log_success, create_progress, console
import time
from rich.panel import Panel
install(show_locals=True)

def main():
    console.print(Panel.fit("[bold blue]奥运会奖牌预测分析系统[/bold blue]", title="系统启动"))

    with create_progress() as progress:
        # 数据加载
        task1 = progress.add_task("[cyan]加载数据...", total=100)
        loader = OlympicsDataLoader()
        data = loader.load_all_data()
        progress.update(task1, completed=100)

        # 特征构建
        task2 = progress.add_task("[green]构建特征...", total=100)
        feature_builder = FeatureBuilder(
            data['medal_counts'],
            data['athletes'],
            data['hosts'],
            data['programs']
        )
        
        features = feature_builder.build_all_features()
        progress.update(task2, completed=100)

        # 显示特征统计信息
        for feature_type, feature_data in features.items():
            console.print(f"\n[yellow]{feature_type}[/yellow] 特征统计:")
            if isinstance(feature_data, dict):
                for name, df in feature_data.items():
                    console.print(f"\n{name}:")
                    console.print(df.describe())
            else:
                console.print(feature_data.describe())
        # 模型训练和预测
        task3 = progress.add_task("[blue]训练模型...", total=100)
        predictor = OlympicsPredictor(features)
        predictor.train_models()
        predictions = predictor.predict_2028()
        progress.update(task3, completed=100)

        # 显示预测结果
        console.print("\n[bold green]2028预测结果:[/bold green]")
        for model_name, pred in predictions.items():
            console.print(f"\n{model_name}模型预测:")
            if model_name == 'gbm':
                console.print(pred.sort_values('predicted_medals', ascending=False).head(10))
            else:
                console.print(pred)

if __name__ == "__main__":
    main()