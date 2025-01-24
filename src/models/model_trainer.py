# src/models/model_trainer.py
import time
from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
import optuna
from neuralprophet import NeuralProphet
from src.models.time_series import ProphetModel

class ModelTrainer:
    def __init__(self, models, features_dict, console=None):
        self.models = models
        self.features = features_dict
        self.console = console or Console()
        self.best_params = {}
        self.model_scores = {}
        
    # def train_and_optimize(self, X, y, model_type='ml'):
    #     """训练和优化模型"""
    #     self.console.print(f"[cyan]开始训练和优化{model_type}模型...[/cyan]")
        
    #     # 数据验证
    #     if len(X) < 10:
    #         raise ValueError(f"数据量不足，当前数据量: {len(X)}，至少需要10个样本。")
        
    #     if model_type == 'dl':
    #         # 确保深度学习输入数据形状正确
    #         if len(X.shape) != 3:
    #             raise ValueError(f"深度学习模型需要3维输入数据，当前维度: {X.shape}。")
        
    #     # 时间序列交叉验证
    #     tscv = TimeSeriesSplit(n_splits=5)
        
    #     # 根据模型类型调用相应的优化方法
    #     if model_type == 'ml':
    #         self._optimize_ml_models(X, y, tscv)
    #     elif model_type == 'dl':
    #         self._optimize_dl_models(X, y, tscv)
    #     elif model_type == 'ts':
    #         self._optimize_ts_models(X, y, tscv)
    #     else:
    #         raise ValueError(f"未知的模型类型: {model_type}，请选择'ml'、'dl'或'ts'。")
        
    #     return self.best_params, self.model_scores
    def train_and_optimize(self, X, y, model_type='ml', max_time=600):
        """训练和优化模型"""
        start_time = time.time()
        results = {}
        scores = {}
        
        try:
            if model_type == 'ts':
                # 时间序列模型训练
                for name, model in self.models.items():
                    if time.time() - start_time > max_time:
                        self.console.print("[yellow]训练时间超过限制，停止后续模型训练[/yellow]")
                        break
                        
                    if name in ['prophet', 'neural_prophet']:
                        try:
                            # 准备数据
                            df = pd.DataFrame({
                                'ds': pd.to_datetime(X.index.astype(str)),
                                'y': y.values
                            })
                            
                            # 训练模型
                            model.fit(df)
                            
                            # 预测
                            future = pd.DataFrame({'ds': pd.to_datetime(X.index.astype(str))})
                            predictions = model.predict(future)
                            
                            # 计算评分
                            if 'yhat' in predictions.columns:
                                pred = predictions['yhat']
                            else:
                                pred = predictions['forecast']
                                
                            scores[name] = mean_squared_error(y, pred)  # 移除squared参数
                            results[name] = model
                            
                        except Exception as e:
                            self.console.print(f"[yellow]警告: 模型 {name} 训练失败: {str(e)}[/yellow]")
                            continue
                            
            elif model_type == 'ml':
                # 机器学习模型训练
                for name, model in self.models.items():
                    if time.time() - start_time > max_time:
                        self.console.print("[yellow]训练时间超过限制，停止后续模型训练[/yellow]")
                        break
                        
                    if name in ['lgb', 'catboost', 'xgboost']:
                        try:
                            model.fit(X, y)
                            pred = model.predict(X)
                            scores[name] = mean_squared_error(y, pred)  # 移除squared参数
                            results[name] = model
                        except Exception as e:
                            self.console.print(f"[yellow]警告: 模型 {name} 训练失败: {str(e)}[/yellow]")
                            continue
            
            elif model_type == 'dl':
                # 深度学习模型训练
                for name, model in self.models.items():
                    if time.time() - start_time > max_time:
                        self.console.print("[yellow]训练时间超过限制，停止后续模型训练[/yellow]")
                        break
                        
                    if name in ['lstm_attention', 'gru', 'tft']:
                        try:
                            history = model.fit(
                                X, y,
                                epochs=50,
                                batch_size=32,
                                verbose=1
                            )
                            pred = model.predict(X)
                            scores[name] = mean_squared_error(y, pred)  # 移除squared参数
                            results[name] = model
                        except Exception as e:
                            self.console.print(f"[yellow]警告: 模型 {name} 训练失败: {str(e)}[/yellow]")
                            continue
            
            self.console.print(f"[green]模型训练完成，用时 {time.time() - start_time:.2f} 秒[/green]")
            return results, scores
            
        except Exception as e:
            self.console.print(f"[red]模型训练过程出错: {str(e)}[/red]")
            return results, scores
    
    def _optimize_ml_models(self, X, y, tscv):
        """优化机器学习模型"""
        for name, model in self.models.items():
            if name in ['lgb', 'catboost', 'xgboost']:
                study = optuna.create_study(direction='minimize')
                
                def objective(trial):
                    params = self._get_ml_params(trial, name)
                    model.set_params(**params)
                    
                    scores = []
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model.fit(X_train, y_train)
                        pred = model.predict(X_val)
                        score = mean_squared_error(y_val, pred)
                        scores.append(score)
                    
                    return np.mean(scores)
                
                study.optimize(objective, n_trials=100)
                self.best_params[name] = study.best_params
                self.model_scores[name] = study.best_value
    
    def _optimize_dl_models(self, X, y, tscv):
        """优化深度学习模型"""
        for name, model in self.models.items():
            if name in ['lstm_attention', 'gru', 'tft']:
                study = optuna.create_study(direction='minimize')
                
                def objective(trial):
                    params = self._get_dl_params(trial)
                    
                    scores = []
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model.fit(
                            X_train, y_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            validation_data=(X_val, y_val),
                            verbose=0
                        )
                        
                        pred = model.predict(X_val)
                        score = mean_squared_error(y_val, pred)
                        scores.append(score)
                    
                    return np.mean(scores)
                
                study.optimize(objective, n_trials=50)
                self.best_params[name] = study.best_params
                self.model_scores[name] = study.best_value
    
    def _optimize_ts_models(self, X, y, tscv):
        """优化时间序列模型"""
        for name, model in self.models.items():
            if name in ['prophet', 'neural_prophet', 'arima_params']:
                study = optuna.create_study(direction='minimize')
                
                def objective(trial):
                    params = self._get_ts_params(trial, name)
                    scores = []
                    
                    for train_idx, val_idx in tscv.split(X):
                        try:
                            if name == 'prophet':
                                # 准备Prophet数据
                                train_df = self._prepare_time_series_data(
                                    X.iloc[train_idx], 
                                    y.iloc[train_idx]
                                )
                                val_df = self._prepare_time_series_data(
                                    X.iloc[val_idx],
                                    y.iloc[val_idx]
                                )
                                
                                # 训练Prophet模型
                                model = Prophet(**params)
                                model.fit(train_df)
                                forecast = model.predict(val_df)
                                score = mean_squared_error(val_df['y'], forecast['yhat'])
                                
                            elif name == 'neural_prophet':
                                # 准备Neural Prophet数据
                                train_df = self._prepare_time_series_data(
                                    X.iloc[train_idx], 
                                    y.iloc[train_idx]
                                )
                                
                                # 训练Neural Prophet模型
                                model = NeuralProphet(**params)
                                model.fit(train_df, freq='Y')
                                future = model.make_future_dataframe(
                                    df=train_df, 
                                    periods=len(val_idx)
                                )
                                forecast = model.predict(future)
                                score = mean_squared_error(
                                    y.iloc[val_idx], 
                                    forecast.tail(len(val_idx))['yhat']
                                )
                                
                            else:  # ARIMA
                                # 准备ARIMA数据
                                train_y = self._prepare_arima_data(y.iloc[train_idx])
                                val_y = self._prepare_arima_data(y.iloc[val_idx])
                                
                                # 训练ARIMA模型
                                model = auto_arima(
                                    train_y,
                                    **{**self.models['arima_params'], **params}
                                )
                                
                                # 预测
                                pred = model.predict(n_periods=len(val_idx))
                                score = mean_squared_error(val_y, pred)
                            
                            scores.append(score)
                            
                        except Exception as e:
                            self.console.print(f"[yellow]警告: 模型 {name} 训练失败: {str(e)}[/yellow]")
                            scores.append(float('inf'))
                    
                    return np.mean(scores) if scores else float('inf')
                
                try:
                    study.optimize(objective, n_trials=50)
                    self.best_params[name] = study.best_params
                    self.model_scores[name] = study.best_value
                except Exception as e:
                    self.console.print(f"[red]模型 {name} 优化失败: {str(e)}[/red]")
    
    def _get_ml_params(self, trial, model_name):
        """获取机器学习模型超参数搜索空间"""
        if model_name == 'lgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
            }
        elif model_name == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0)
            }
        else:  # xgboost
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
            }
    
    def _get_dl_params(self, trial):
        """获取深度学习模型超参数搜索空间"""
        return {
            'epochs': trial.suggest_int('epochs', 50, 200),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        }
    
    def _get_ts_params(self, trial, model_name):
        """获取时间序列模型超参数搜索空间"""
        if model_name == 'prophet':
            return {
                'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5),
                'seasonality_prior_scale': trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10.0)
            }
        elif model_name == 'neural_prophet':
            return {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
            }
        else:  # arima
            return {
                'max_p': trial.suggest_int('max_p', 1, 5),
                'max_d': trial.suggest_int('max_d', 0, 2),
                'max_q': trial.suggest_int('max_q', 1, 5)
            }
    # def _prepare_time_series_data(self, X, y):
    #     """准备时间序列数据"""
    #     # 确保数据是有序的
    #     df = pd.DataFrame({
    #         'ds': X.index,
    #         'y': y.values
    #     }).sort_values('ds')
        
    #     # 处理缺失值
    #     df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
        
    #     # 确保日期格式正确
    #     df['ds'] = pd.to_datetime(df['ds'])
        
    #     return df
    def _prepare_time_series_data(self, X, y):
        """准备时间序列数据"""
        try:
            # 创建时间索引
            if isinstance(X.index, pd.DatetimeIndex):
                dates = X.index
            else:
                dates = pd.to_datetime(X.index.astype(str))
            
            # 构建数据框
            df = pd.DataFrame({
                'ds': dates,
                'y': y.values
            })
            
            # 排序并处理缺失值
            df = df.sort_values('ds')
            df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            self.console.print(f"[red]时间序列数据准备失败: {str(e)}[/red]")
            raise
    def _prepare_arima_data(self, y):
        """准备ARIMA数据"""
        # 确保数据是一维数组
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.ravel()
        
        # 处理缺失值
        y = pd.Series(y).fillna(method='ffill').fillna(method='bfill').values
        
        return y