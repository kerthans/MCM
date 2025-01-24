# src/models/predictor.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
import scipy.stats as stats

class OlympicPredictor:
    def __init__(self, trained_models, features, console=None):
        self.models = trained_models
        self.features = features
        self.console = console or Console()
        self.predictions = {}
        self.confidence_intervals = {}
        
    def predict_2028(self, X_future):
        """生成2028年预测"""
        self.console.print("[cyan]生成2028年预测...[/cyan]")
        
        predictions = {}
        confidence_intervals = {}
        
        # 各模型预测
        for name, model in self.models.items():
            try:
                if name in ['prophet', 'neural_prophet']:
                    pred = self._predict_prophet(model, X_future)
                else:
                    pred = model.predict(X_future)
                
                predictions[name] = pred
                
                # 计算预测区间
                if hasattr(model, 'predict_proba'):
                    lower, upper = self._calculate_prediction_interval(model, X_future)
                    confidence_intervals[name] = (lower, upper)
                
            except Exception as e:
                self.console.print(f"[red]模型 {name} 预测失败: {str(e)}[/red]")
        
        # 集成预测
        ensemble_pred = self._ensemble_predictions(predictions)
        ensemble_ci = self._ensemble_confidence_intervals(confidence_intervals)
        
        self.predictions = predictions
        self.confidence_intervals = confidence_intervals
        
        return ensemble_pred, ensemble_ci
    
    def analyze_host_effect(self):
        """分析主办国效应"""
        host_effect = pd.DataFrame()
        
        # 计算历史主办国效应
        historical_hosts = self.features['hosts']
        medal_counts = self.features['medal_counts']
        
        for year in historical_hosts['Year'].unique():
            host = historical_hosts[historical_hosts['Year'] == year]['Host'].iloc[0]
            before_avg = medal_counts[
                (medal_counts['NOC'] == host) & 
                (medal_counts['Year'] < year)
            ]['Total'].mean()
            
            during = medal_counts[
                (medal_counts['NOC'] == host) & 
                (medal_counts['Year'] == year)
            ]['Total'].iloc[0]
            
            after_avg = medal_counts[
                (medal_counts['NOC'] == host) & 
                (medal_counts['Year'] > year)
            ]['Total'].mean()
            
            host_effect.loc[year, 'Host'] = host
            host_effect.loc[year, 'Before_Avg'] = before_avg
            host_effect.loc[year, 'During'] = during
            host_effect.loc[year, 'After_Avg'] = after_avg
            host_effect.loc[year, 'Effect'] = during - before_avg
        
        return host_effect
    
    def analyze_coach_effect(self):
        """分析教练效应"""
        # 这里需要根据实际数据结构进行调整
        coach_effect = pd.DataFrame()
        
        # 计算教练影响
        if 'Coach' in self.features['athletes'].columns:
            coach_data = self.features['athletes'].groupby(['Coach', 'NOC', 'Year'])['Medal'].count().reset_index()
            
            for coach in coach_data['Coach'].unique():
                before_coach = coach_data[coach_data['Coach'] != coach].groupby('NOC')['Medal'].mean()
                with_coach = coach_data[coach_data['Coach'] == coach].groupby('NOC')['Medal'].mean()
                
                effect = with_coach - before_coach
                coach_effect.loc[coach, 'Average_Effect'] = effect.mean()
                coach_effect.loc[coach, 'Effect_Std'] = effect.std()
        
        return coach_effect
    
    def visualize_results(self):
        """可视化预测结果"""
        # 预测趋势图
        plt.figure(figsize=(12, 6))
        for name, pred in self.predictions.items():
            plt.plot(pred, label=name)
        plt.title('2028奥运会预测趋势')
        plt.legend()
        plt.show()
        
        # 主办国效应图
        host_effect = self.analyze_host_effect()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=host_effect, x='Host', y='Effect')
        plt.title('主办国效应分析')
        plt.xticks(rotation=45)
        plt.show()
        
        # 特征重要性图
        if hasattr(self.models['lgb'], 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.features.columns,
                'importance': self.models['lgb'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance.head(20), x='importance', y='feature')
            plt.title('特征重要性分析')
            plt.show()
    
    def _predict_prophet(self, model, future_dates):
        """Prophet模型预测"""
        future = model.make_future_dataframe(periods=1, freq='Y')
        forecast = model.predict(future)
        return forecast.iloc[-1]['yhat']
    
    def _calculate_prediction_interval(self, model, X, confidence=0.95):
        """计算预测区间"""
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            lower = np.percentile(proba, (1 - confidence) * 100 / 2)
            upper = np.percentile(proba, (1 + confidence) * 100 / 2)
            return lower, upper
        else:
            pred = model.predict(X)
            std = np.std(pred)
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin = z_score * std
            return pred - margin, pred + margin
    
    def _ensemble_predictions(self, predictions):
        """集成多个模型的预测"""
        # 使用加权平均
        weights = {
            'lgb': 0.3,
            'catboost': 0.3,
            'xgboost': 0.2,
            'prophet': 0.1,
            'neural_prophet': 0.1
        }
        
        weighted_pred = 0
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in weights:
                weighted_pred += pred * weights[name]
                total_weight += weights[name]
        
        return weighted_pred / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
    
    def _ensemble_confidence_intervals(self, intervals):
        """集成多个模型的置信区间"""
        all_lower = []
        all_upper = []
        
        for lower, upper in intervals.values():
            all_lower.append(lower)
            all_upper.append(upper)
        
        return np.mean(all_lower), np.mean(all_upper)