# src/models/base_models.py
import pandas as pd
import numpy as np
# from prophet import Prophet
try:
    from prophet import Prophet
except ImportError:
    print("Warning: Prophet not installed, some models will be unavailable")
    Prophet = None

try:
    from neuralprophet import NeuralProphet
except ImportError:
    print("Warning: NeuralProphet not installed, some models will be unavailable")
    NeuralProphet = None

# from neuralprophet import NeuralProphet
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from rich.console import Console

class OlympicModelBuilder:
    def __init__(self, features_dict, console=None):
        self.features = features_dict
        self.console = console or Console()
        self.models = {}
        self.scalers = {}
        self.histories = {}
        
        # 初始化模型输入维度
        self.input_dim = self._calculate_input_dim()
        self.sequence_length = 10  # 设置时间序列长度
        
    # def _calculate_input_dim(self):
    #     """计算模型输入维度"""
    #     try:
    #         # 如果features是字典类型
    #         if isinstance(self.features, dict):
    #             # 获取time_series特征的维度
    #             if 'time_series' in self.features and isinstance(self.features['time_series'], pd.DataFrame):
    #                 return self.features['time_series'].shape[1]
                    
    #             # 获取第一个可用特征集的维度
    #             for feature_set in self.features.values():
    #                 if isinstance(feature_set, pd.DataFrame):
    #                     return feature_set.shape[1]
    #                 elif isinstance(feature_set, np.ndarray):
    #                     return feature_set.shape[1] if len(feature_set.shape) > 1 else 1
            
    #         # 如果是单个DataFrame
    #         elif isinstance(self.features, pd.DataFrame):
    #             return self.features.shape[1]
                
    #         # 如果是单个numpy数组
    #         elif isinstance(self.features, np.ndarray):
    #             return self.features.shape[1] if len(self.features.shape) > 1 else 1
                
    #         # 如果无法确定维度，返回当前输入维度
    #         return 2109
            
    #     except Exception as e:
    #         self.console.print(f"[yellow]警告: 输入维度计算失败，使用默认值: {str(e)}[/yellow]")
    #         return 2109
    def _calculate_input_dim(self):
        """计算模型输入维度"""
        try:
            # 如果features是字典类型
            if isinstance(self.features, dict):
                # 获取第一个可用特征集的维度
                for feature_set in self.features.values():
                    if isinstance(feature_set, pd.DataFrame):
                        return feature_set.shape[1]
                    elif isinstance(feature_set, np.ndarray):
                        return feature_set.shape[1] if len(feature_set.shape) > 1 else 1
                        
            # 如果是单个DataFrame
            elif isinstance(self.features, pd.DataFrame):
                return self.features.shape[1]
                
            # 如果是单个numpy数组
            elif isinstance(self.features, np.ndarray):
                return self.features.shape[1] if len(self.features.shape) > 1 else 1
                
            raise ValueError("无法确定输入维度")
                
        except Exception as e:
            self.console.print(f"[yellow]警告: 输入维度计算失败: {str(e)}[/yellow]")
            return 184  # 使用预处理后的默认维度
    def build_time_series_models(self):
        """构建时间序列模型组"""
        self.console.print("[cyan]构建时间序列模型...[/cyan]")
        
        try:
            models = {}
            
            # Prophet模型
            if Prophet is not None:
                prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    interval_width=0.95
                )
                models['prophet'] = prophet_model
            
            # Neural Prophet模型
            if NeuralProphet is not None:
                neural_prophet = NeuralProphet(
                    growth="linear",          # 线性增长
                    yearly_seasonality=True,   # 年度季节性
                    weekly_seasonality=False,  # 关闭周季节性
                    daily_seasonality=False,   # 关闭日季节性
                    learning_rate=0.01,
                    batch_size=32,
                    epochs=50,
                    loss_func='MSE',
                    normalize='standardize',
                    n_forecasts=1,
                    n_lags=10,
                    seasonality_mode='multiplicative',
                    trend_reg=0.1,
                    ar_reg=0.1
                )
                models['neural_prophet'] = neural_prophet
            
            # ARIMA参数
            arima_params = {
                'start_p': 1,
                'start_q': 1,
                'max_p': 3,
                'max_q': 3,
                'max_d': 2,
                'm': 1,  # 年度数据
                'seasonal': False,  # 关闭季节性
                'trace': True,
                'error_action': 'ignore',
                'suppress_warnings': True,
                'stepwise': True
            }
            models['arima_params'] = arima_params
            
            self.models.update(models)
            self.console.print("[green]✓ 时间序列模型构建完成[/green]")
            
        except Exception as e:
            self.console.print(f"[red]时间序列模型构建失败: {str(e)}[/red]")
            raise
            
        return self.models
    # def build_time_series_models(self):
    #     """构建时间序列模型组"""
    #     self.console.print("[cyan]构建时间序列模型...[/cyan]")
        
    #     try:
    #         # Prophet模型
    #         prophet_model = Prophet(
    #             yearly_seasonality=True,
    #             weekly_seasonality=False,
    #             daily_seasonality=False,
    #             changepoint_prior_scale=0.05,
    #             interval_width=0.95
    #         )
            
    #         # Neural Prophet模型
    #         neural_prophet = NeuralProphet(
    #             yearly_seasonality=True,
    #             weekly_seasonality=False,
    #             daily_seasonality=False,
    #             learning_rate=0.01,
    #             batch_size=32,
    #             epochs=100,
    #             loss_func='MSE'
    #         )
            
    #         # ARIMA参数
    #         arima_params = {
    #             'start_p': 1,
    #             'start_q': 1,
    #             'max_p': 3,
    #             'max_q': 3,
    #             'start_P': 0,
    #             'start_Q': 0,
    #             'max_P': 2,
    #             'max_Q': 2,
    #             'm': 4,
    #             'seasonal': True,
    #             'trace': True,
    #             'error_action': 'ignore',
    #             'suppress_warnings': True,
    #             'stepwise': True
    #         }
            
    #         self.models.update({
    #             'prophet': prophet_model,
    #             'neural_prophet': neural_prophet,
    #             'arima_params': arima_params
    #         })
            
    #         self.console.print("[green]✓ 时间序列模型构建完成[/green]")
            
    #     except Exception as e:
    #         self.console.print(f"[red]时间序列模型构建失败: {str(e)}[/red]")
    #         raise
            
    #     return self.models
    
    def build_ml_models(self):
        """构建机器学习模型组"""
        self.console.print("[cyan]构建机器学习模型...[/cyan]")
        
        try:
            # LightGBM模型
            lgb_model = LGBMRegressor(
                objective='regression',
                n_estimators=1000,
                learning_rate=0.01,
                num_leaves=31,
                min_child_samples=5,  # 降低最小样本要求
                min_data_in_leaf=5,   # 降低叶子节点最小样本要求
                min_data_in_bin=5,    # 降低分箱最小样本要求
                random_state=42,
                verbose=-1            # 减少警告输出
            )
            
            # CatBoost模型
            cat_model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.01,
                depth=4,              # 降低树深度
                min_data_in_leaf=5,   # 降低最小样本要求
                random_seed=42,
                verbose=False
            )
            
            # XGBoost模型
            xgb_model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=4,          # 降低树深度
                min_child_weight=3,   # 降低最小样本权重
                random_state=42,
                verbosity=0          # 减少警告输出
            )
            
            self.models.update({
                'lgb': lgb_model,
                'catboost': cat_model,
                'xgboost': xgb_model
            })
            
            self.console.print("[green]✓ 机器学习模型构建完成[/green]")
            
        except Exception as e:
            self.console.print(f"[red]机器学习模型构建失败: {str(e)}[/red]")
            raise
            
        return self.models
    
    def build_deep_learning_models(self):
        """构建深度学习模型组"""
        self.console.print("[cyan]构建深度学习模型...[/cyan]")
        
        try:
            # 重新计算并更新输入维度
            self.input_dim = self._calculate_input_dim()
            self.console.print(f"[cyan]当前输入维度: {self.input_dim}[/cyan]")
            
            # 构建模型
            models = {}
            
            # 清除之前的模型
            tf.keras.backend.clear_session()
            
            try:
                models['lstm_attention'] = self._build_lstm_attention_model()
                self.console.print("[green]✓ LSTM-Attention模型构建完成[/green]")
            except Exception as e:
                self.console.print(f"[yellow]LSTM-Attention模型构建失败: {str(e)}[/yellow]")
            
            try:
                models['gru'] = self._build_gru_model()
                self.console.print("[green]✓ GRU模型构建完成[/green]")
            except Exception as e:
                self.console.print(f"[yellow]GRU模型构建失败: {str(e)}[/yellow]")
            
            try:
                models['tft'] = self._build_tft_model()
                self.console.print("[green]✓ TFT模型构建完成[/green]")
            except Exception as e:
                self.console.print(f"[yellow]TFT模型构建失败: {str(e)}[/yellow]")
            
            self.models.update(models)
            
            if not models:
                raise ValueError("没有成功构建任何深度学习模型")
            
            self.console.print("[green]✓ 深度学习模型构建完成[/green]")
            return self.models
            
        except Exception as e:
            self.console.print(f"[red]深度学习模型构建失败: {str(e)}[/red]")
            return self.models

    def _build_lstm_attention_model(self):
        """构建LSTM-Attention模型"""
        try:
            # 获取当前输入维度
            current_dim = self._calculate_input_dim()
            self.console.print(f"[cyan]LSTM-Attention模型输入维度: {current_dim}[/cyan]")
            
            inputs = Input(shape=(self.sequence_length, current_dim))
            
            # 降低模型复杂度
            lstm_out = LSTM(64, return_sequences=True)(inputs)
            attention = MultiHeadAttention(num_heads=2, key_dim=32)(lstm_out, lstm_out)
            
            x = tf.keras.layers.Add()([attention, lstm_out])
            x = LayerNormalization()(x)
            
            x = GlobalAveragePooling1D()(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.1)(x)
            outputs = Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            self.console.print(f"[red]LSTM-Attention模型构建失败: {str(e)}[/red]")
            raise

    def _build_gru_model(self):
        """构建GRU模型"""
        try:
            current_dim = self._calculate_input_dim()
            self.console.print(f"[cyan]GRU模型输入维度: {current_dim}[/cyan]")
            
            inputs = Input(shape=(self.sequence_length, current_dim))
            
            x = GRU(32, return_sequences=True)(inputs)
            x = GRU(16)(x)
            x = Dense(8, activation='relu')(x)
            x = Dropout(0.1)(x)
            outputs = Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            self.console.print(f"[red]GRU模型构建失败: {str(e)}[/red]")
            raise

    def _build_tft_model(self):
        """构建简化版Temporal Fusion Transformer"""
        try:
            current_dim = self._calculate_input_dim()
            self.console.print(f"[cyan]TFT模型输入维度: {current_dim}[/cyan]")
            
            inputs = Input(shape=(self.sequence_length, current_dim))
            
            attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
            x = tf.keras.layers.Add()([attention, inputs])
            x = LayerNormalization()(x)
            
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            x = GlobalAveragePooling1D()(x)
            outputs = Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            self.console.print(f"[red]TFT模型构建失败: {str(e)}[/red]")
            raise

    # def prepare_sequence_data(self, X, y, min_samples=10):
    #     """准备序列数据"""
    #     # 输入验证
    #     if len(X) != len(y):
    #         raise ValueError("输入特征和目标值长度不匹配")
        
    #     # 确保索引是数值类型
    #     if isinstance(X, pd.DataFrame):
    #         X.index = pd.to_numeric(X.index, errors='coerce')
    #         X = X.dropna(subset=[X.index.name or 'index'])
    #         X.index = X.index.astype(int)
        
    #     if isinstance(y, pd.Series):
    #         y.index = pd.to_numeric(y.index, errors='coerce')
    #         y = y.dropna()
    #         y.index = y.index.astype(int)
        
    #     # 动态调整序列长度
    #     self.sequence_length = min(self.sequence_length, len(X) // 3)
    #     if self.sequence_length < 2:
    #         self.sequence_length = 2
        
    #     # 确保有足够的数据
    #     if len(X) < min_samples:
    #         raise ValueError(f"数据量不足，至少需要{min_samples}个样本")
        
    #     sequences = []
    #     targets = []
        
    #     # 使用滑动窗口创建序列
    #     for i in range(len(X) - self.sequence_length):
    #         if isinstance(X, pd.DataFrame):
    #             seq = X.iloc[i:(i + self.sequence_length)].values
    #         else:
    #             seq = X[i:(i + self.sequence_length)]
                
    #         if isinstance(y, pd.Series):
    #             target = y.iloc[i + self.sequence_length]
    #         else:
    #             target = y[i + self.sequence_length]
                
    #         sequences.append(seq)
    #         targets.append(target)
        
    #     # 转换为numpy数组
    #     sequences = np.array(sequences)
    #     targets = np.array(targets)
        
    #     # 打印序列形状信息
    #     self.console.print(f"[cyan]序列数据形状: {sequences.shape}[/cyan]")
    #     self.console.print(f"[cyan]目标数据形状: {targets.shape}[/cyan]")
        
    #     return sequences, targets
    def prepare_sequence_data(self, X, y, min_samples=10):
        """准备序列数据"""
        try:
            if len(X) != len(y):
                raise ValueError("输入特征和目标值长度不匹配")
            
            # 特征预处理
            X = self.preprocess_features(X)
            self.console.print(f"[cyan]预处理后特征维度: {X.shape}[/cyan]")
            
            # 更新输入维度
            self.input_dim = X.shape[1]
            
            # 清除现有模型
            tf.keras.backend.clear_session()
            
            # 用新维度重建模型
            self.models.update({
                'lstm_attention': self._build_lstm_attention_model(),
                'gru': self._build_gru_model(),
                'tft': self._build_tft_model()
            })
            
            # 转换输入数据
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
            
            # 确保索引是数值类型
            if not X.index.is_numeric():
                X.index = pd.RangeIndex(len(X))
            if not y.index.is_numeric():
                y.index = pd.RangeIndex(len(y))
            
            # 调整序列长度
            self.sequence_length = min(self.sequence_length, len(X) // 3)
            self.sequence_length = max(2, self.sequence_length)
            
            if len(X) < min_samples:
                raise ValueError(f"数据量不足，至少需要{min_samples}个样本")
            
            sequences = []
            targets = []
            
            for i in range(len(X) - self.sequence_length):
                seq = X.iloc[i:(i + self.sequence_length)].values
                target = y.iloc[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            self.console.print(f"[cyan]序列数据形状: {sequences.shape}[/cyan]")
            self.console.print(f"[cyan]目标数据形状: {targets.shape}[/cyan]")
            
            return sequences, targets
            
        except Exception as e:
            self.console.print(f"[red]序列数据准备失败: {str(e)}[/red]")
            raise
    def preprocess_features(self, X):
        """预处理特征数据"""
        try:
            if isinstance(X, pd.DataFrame):
                # 删除常量列
                constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
                if constant_cols:
                    X = X.drop(columns=constant_cols)
                    self.console.print(f"[yellow]删除 {len(constant_cols)} 个常量特征[/yellow]")
                
                # 处理缺失值
                X = X.fillna(X.mean())
                
                # 处理无限值
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.mean())
                
                # 删除高度相关特征
                corr_matrix = X.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
                if high_corr_cols:
                    X = X.drop(columns=high_corr_cols)
                    self.console.print(f"[yellow]删除 {len(high_corr_cols)} 个高度相关特征[/yellow]")
                
                # 标准化数值特征
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                
                self.console.print(f"[green]特征预处理完成，最终特征维度: {X_scaled.shape}[/green]")
                return X_scaled
                
            return X
            
        except Exception as e:
            self.console.print(f"[red]特征预处理失败: {str(e)}[/red]")
            return X