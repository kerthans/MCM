# src/models/predictor.py
import pandas as pd
import numpy as np
from src.utils.logger import log_info, log_success
from src.models.time_series import ProphetModel
from src.models.ml_models import GBMModel

class OlympicsPredictor:
   def __init__(self, features):
       self.features = features
       self.models = {}
   def prepare_training_data(self, target_col='Total'):
       # 获取特征
       country_stats = self.features['country']['country_stats'].copy()
       recent_perf = self.features['country']['recent_performance'].copy()
       sport_div = pd.DataFrame(self.features['sport']['sport_diversity']).copy()
       
       # 重命名列
       country_stats.columns = [
           'Gold_mean', 'Gold_std', 'Gold_max', 'Gold_sum',
           'Total_mean', 'Total_std', 'Total_max', 'Total_sum', 
           'Year_count'
       ]
       
       recent_perf.columns = ['Recent_Gold', 'Recent_Total']
       sport_div.columns = ['Sport_Diversity']

       # 合并特征
       df = pd.concat([
           country_stats,
           recent_perf,
           sport_div
       ], axis=1)

       # 处理缺失值
       df = df.fillna(0)
       
       # 确保列名一致性
       df.columns = [col.replace(' ', '_') for col in df.columns]
       
       return df
       
   def train_models(self):
       X = self.prepare_training_data()
       y = X['Recent_Total']  # 使用最近的总奖牌数作为目标
       
       # 特征列
       self.feature_cols = [col for col in X.columns 
                          if col not in ['Recent_Total', 'Recent_Gold']]
       X = X[self.feature_cols]

       # GBM模型训练
       log_info("训练GBM模型...")
       gbm = GBMModel()
       gbm.fit(X, y)
       self.models['gbm'] = gbm
       
       log_success("模型训练完成")

   def predict_2028(self):
       X_pred = self.prepare_training_data()
       X_pred = X_pred[self.feature_cols]  # 使用训练时的特征列

       # GBM预测
       gbm_pred = self.models['gbm'].predict(X_pred)
       predictions = pd.DataFrame({
           'NOC': X_pred.index,
           'predicted_medals_2028': gbm_pred
       }).sort_values('predicted_medals_2028', ascending=False)

       return predictions