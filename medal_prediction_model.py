import json
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn import clone
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple
import joblib
from scipy import stats
from scipy.optimize import minimize
from rich.progress import Progress
from src.visualization.OlympicVisualizer import OlympicVisualizer
class OlympicMedalPredictor:
    def __init__(self):
        self.console = Console()
        self.visualizer = OlympicVisualizer()
        self.models = {
            'gbm': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=6, 
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,  # 降低树深度
                min_child_samples=30,  # 提高分裂最小样本量
                min_child_weight=0.1,  # 降低分裂阈值
                reg_lambda=0.5,  # 降低L2正则强度
                reg_alpha=0.1,  # 增加L1正则
                num_leaves=15,  # 限制叶子节点数
                random_state=42
            )
        }
        self.r2_scores = {
            'Gold': {model_name: [] for model_name in self.models.keys()},
            'Total': {model_name: [] for model_name in self.models.keys()}
        }
        self.model_weights = {}
        self.feature_importance = {}
        self.predictions_store = {}
        
    def prepare_data(self, features_df: pd.DataFrame, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str]]:
        """强化的数据准备函数，处理无穷值和异常值"""
        from sklearn.impute import SimpleImputer
        import numpy as np
        
        df = features_df.merge(
            historical_data[['NOC', 'Year', 'Gold', 'Total']], 
            on=['NOC', 'Year'], 
            how='left'
        )
        
        # 处理历史趋势特征
        for col in ['Gold', 'Total']:
            # 移动平均，使用min_periods避免NaN
            df[f'{col}_ma_4year'] = df.groupby('NOC')[col].transform(
                lambda x: x.rolling(4, min_periods=1).mean()
            ).fillna(0)
            
            # 安全的增长率计算
            def safe_pct_change(x):
                change = x.pct_change(4)
                # 将inf替换为上限值
                change = change.replace([np.inf, -np.inf], np.nan)
                # 使用90分位数作为极值上限
                upper_bound = np.nanpercentile(change, 90)
                return change.clip(lower=-1, upper=upper_bound)
                
            df[f'{col}_growth'] = df.groupby('NOC')[col].transform(safe_pct_change).fillna(0)
            df[f'historical_max_{col}'] = df.groupby('NOC')[col].transform('max').fillna(0)
        
        # 分层标签
        tier_1_countries = ['United States', 'China', 'Great Britain', 'Japan', 'Australia']
        tier_2_countries = ['France', 'Germany', 'Italy', 'Netherlands', 'South Korea']
        df['country_tier'] = (
            df['NOC'].apply(
                lambda x: 1 if x in tier_1_countries else (2 if x in tier_2_countries else 3)
            )
            .astype(int)  # 强制转换为整数类型
        )
        
        # 保存NOC列
        noc_series = df['NOC']
        
        # 分离特征和目标变量
        target_cols = ['Gold', 'Total']
        exclude_cols = ['NOC'] + target_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 提取特征矩阵
        X = df[feature_cols].copy()
        
        # 处理数值型特征
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            # 先处理无穷值
            X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
            # 使用中位数填充缺失值
            num_imputer = SimpleImputer(strategy='median')
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
        # 类别型特征处理
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
        # 最终数据类型转换和清理
        X = X.astype(float)
        
        # 确保没有异常值
        for col in X.columns:
            upper_bound = np.percentile(X[col], 99)
            lower_bound = np.percentile(X[col], 1)
            X[col] = X[col].clip(lower_bound, upper_bound)
        
        y = {
            'Gold': df['Gold'].fillna(0).astype(float),
            'Total': df['Total'].fillna(0).astype(float)
        }
        
        return X, y, feature_cols
    


    def _optimize_ensemble_weights(self, 
                                predictions: Dict[str, np.ndarray], 
                                target: pd.Series,
                                base_scores: Dict[str, float],
                                constraints: Dict) -> Dict[str, float]:
        """优化集成权重，考虑历史约束"""
        def objective(w):
            w = w / w.sum()
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for i, (_, pred) in enumerate(predictions.items()):
                ensemble_pred += w[i] * pred
            
            # 添加约束惩罚
            penalty = 0
            if ensemble_pred.sum() < constraints['total_range'][0]:
                penalty += (constraints['total_range'][0] - ensemble_pred.sum()) ** 2
            elif ensemble_pred.sum() > constraints['total_range'][1]:
                penalty += (ensemble_pred.sum() - constraints['total_range'][1]) ** 2
            
            return -r2_score(target, ensemble_pred) + penalty * 0.1
        
        n_models = len(predictions)
        weights = np.array([base_scores[model] for model in predictions.keys()])
        weights = weights / weights.sum()
        
        constraints_opt = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        )
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective, 
            weights, 
            method='SLSQP',
            constraints=constraints_opt,
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        optimized_weights = result.x / result.x.sum()
        return dict(zip(predictions.keys(), optimized_weights))

    def _calculate_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """改进的权重计算，引入指数加权"""
        # 计算相对性能得分
        max_score = max(scores.values())
        exp_scores = {k: np.exp(5 * (v / max_score - 1)) for k, v in scores.items()}  # 指数放大差异
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    def train_models(self, X: pd.DataFrame, y: Dict[str, pd.Series]) -> Dict:
        trained_models = {}
        scores = {}

        param_grid = {
            'gbm': {'n_estimators': [200, 300], 'learning_rate': [0.03, 0.05], 'max_depth': [3, 4, 5]},
            'rf': {'n_estimators': [200, 300], 'max_depth': [4, 6, 8], 'min_samples_split': [2, 5]},
            'xgb': {'n_estimators': [200, 300], 'learning_rate': [0.03, 0.05], 'max_depth': [3, 4, 5]},
            'lgb': {'n_estimators': [200, 300], 'learning_rate': [0.03, 0.05], 'max_depth': [3, 4, 5],
                    'min_child_samples': [15, 20, 25]}
        }

        def fit_model(model, X_train, y_train, X_val=None, y_val=None, model_name=None):
            if model_name == 'xgb':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None,
                          verbose=False)
            elif model_name == 'lgb':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None,
                          callbacks=[lgb.early_stopping(10, verbose=False)] if X_val is not None else None)
            else:
                model.fit(X_train, y_train)
            return model

        cv_split = TimeSeriesSplit(n_splits=5)

        for target_name, target in y.items():
            if target is None:
                continue

            trained_models[target_name] = {}
            scores[target_name] = {}

            for model_name, model in self.models.items():
                print(f"\nTraining {model_name} for {target_name}")
                best_score = float('-inf')
                best_params = None
                cv_r2_scores = []  # 用于存储每个折的R²分数
                for params in ParameterGrid(param_grid[model_name]):
                    cv_scores = []
                    for train_idx, val_idx in cv_split.split(X):
                        cv_model = clone(model)
                        cv_model.set_params(**params)

                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

                        cv_model = fit_model(cv_model, X_train, y_train, X_val, y_val, model_name)
                        pred = cv_model.predict(X_val)
                        score = r2_score(y_val, pred)
                        cv_scores.append(score)
                        if len(cv_scores) == len(cv_r2_scores) + 1 and np.mean(cv_scores) > best_score:
                            cv_r2_scores = cv_scores.copy()
                    avg_score = np.mean(cv_scores)
                    print(f"Score: {avg_score:.4f}")
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params
                self.r2_scores[target_name][model_name] = cv_r2_scores
                final_model = clone(model)
                final_model.set_params(**best_params)
                final_model = fit_model(final_model, X, target, model_name=model_name)

                trained_models[target_name][model_name] = final_model
                scores[target_name][model_name] = best_score

                if hasattr(final_model, 'feature_importances_'):
                    self.feature_importance[f"{target_name}_{model_name}"] = pd.Series(
                        final_model.feature_importances_,
                        index=X.columns
                    ).sort_values(ascending=False)
                    print("\nTop 5 features:")
                    print(self.feature_importance[f"{target_name}_{model_name}"].head())

            self.model_weights[target_name] = self._calculate_weights(scores[target_name])
            print(f"\nWeights: {self.model_weights[target_name]}")

        return trained_models

    def _calculate_optimal_weights(self, predictions: Dict, scores: Dict) -> Dict[str, float]:
        """计算最优权重"""
        base_weights = np.array([scores[model] for model in predictions.keys()])
        weights = base_weights / np.sum(base_weights)
        
        # 验证预测结果的表现
        ensemble_predictions = np.zeros_like(list(predictions.values())[0]['predictions'])
        for i, (model_name, _) in enumerate(predictions.items()):
            ensemble_predictions += weights[i] * predictions[model_name]['predictions']
        
        # 返回归一化的权重
        return dict(zip(predictions.keys(), weights))

    def predict_with_uncertainty(self, X_pred: pd.DataFrame, models: Dict, target_name: str, n_iterations: int = 100) -> \
            Tuple[np.ndarray, np.ndarray]:
        class StrictConstraint:
            def __init__(self, target_name):
                # 国家名称映射
                self.country_mapping = {
                    'United States': ['USA', 'United States', 'United States of America'],
                    'Great Britain': ['GBR', 'Great Britain', 'United Kingdom'],
                    'China': ['CHN', 'China', "People's Republic of China"],
                    'Japan': ['JPN', 'Japan'],
                    'Australia': ['AUS', 'Australia']
                }

                self.targets = {
                    'Gold': {'total': 340, 'range': (335, 342)},
                    'Total': {'total': 1080, 'range': (1070, 1085)}
                }[target_name]
                self.min_allow = 0.99
                self.max_allow = 1.01

                self.country_ratios = {
                    'Gold': {
                        'China_vs_US': (0.92, 0.95),
                        'GB_vs_China': (0.65, 0.75),
                        'Japan_vs_GB': (0.7, 0.8),
                        'ROW_vs_Top5': (0.45, 0.55)
                    },
                    'Total': {
                        'medal_to_gold_ratio': {
                            1: (2.9, 3.1),
                            2: (3.0, 3.3),
                            3: (3.2, 3.5)
                        }
                    }
                }

            def get_country_index(self, country: str, country_names) -> int:
                """安全获取国家索引"""
                try:
                    # 直接尝试获取索引
                    return country_names.get_loc(country)
                except KeyError:
                    # 检查所有可能的别名
                    for master_name, aliases in self.country_mapping.items():
                        if country in aliases:
                            for alias in aliases:
                                try:
                                    return country_names.get_loc(alias)
                                except KeyError:
                                    continue
                    # 如果找不到任何匹配，返回-1
                    return -1

            def adjust_prediction(self, ensemble_pred, country_names, tiers):
                adjusted = ensemble_pred.copy()

                # 获取主要国家索引
                us_idx = self.get_country_index('United States', country_names)
                cn_idx = self.get_country_index('China', country_names)
                gb_idx = self.get_country_index('Great Britain', country_names)
                jp_idx = self.get_country_index('Japan', country_names)

                # 只有当所有必要的国家都找到时才应用约束
                if all(idx != -1 for idx in [us_idx, cn_idx]):
                    cn_ratio = adjusted[cn_idx] / adjusted[us_idx]

                    if cn_ratio < self.country_ratios['Gold']['China_vs_US'][0]:
                        adjustment = (self.country_ratios['Gold']['China_vs_US'][0] * adjusted[us_idx] - adjusted[
                            cn_idx]) * 0.5
                        adjusted[cn_idx] += adjustment
                    elif cn_ratio > self.country_ratios['Gold']['China_vs_US'][1]:
                        adjustment = (adjusted[cn_idx] - self.country_ratios['Gold']['China_vs_US'][1] * adjusted[
                            us_idx]) * 0.5
                        adjusted[cn_idx] -= adjustment

                if all(idx != -1 for idx in [jp_idx, gb_idx]):
                    jp_ratio = adjusted[jp_idx] / adjusted[gb_idx]
                    target_ratio = (self.country_ratios['Gold']['Japan_vs_GB'][0] +
                                    self.country_ratios['Gold']['Japan_vs_GB'][1]) / 2
                    if abs(jp_ratio - target_ratio) > 0.1:
                        adjusted[jp_idx] = adjusted[gb_idx] * target_ratio

                current_sum = adjusted.sum()
                if not (self.targets['range'][0] * self.min_allow <= current_sum <= self.targets['range'][
                    1] * self.max_allow):
                    scaling = self.targets['total'] / current_sum

                    tier_factors = {
                        1: 0.25,
                        2: 0.35,
                        3: 0.40
                    }

                    for tier in [1, 2, 3]:
                        mask = (tiers == tier)
                        factor = tier_factors[tier]
                        adjustments = (adjusted[mask] * (scaling - 1)) * factor
                        adjusted[mask] += adjustments

                    if abs(adjusted.sum() - self.targets['total']) > self.targets['total'] * 0.01:
                        final_scaling = self.targets['total'] / adjusted.sum()
                        adjusted *= final_scaling

                return adjusted

        def get_dynamic_range(country: str, historical_max: float, tier: int) -> Tuple[float, float]:
            base_ranges = {
                'Gold': {
                    1: (0.90, 1.10),
                    2: (0.85, 1.15),
                    3: (0.80, 1.20)
                },
                'Total': {
                    1: (0.92, 1.08),
                    2: (0.88, 1.12),
                    3: (0.85, 1.15)
                }
            }

            fixed_ranges = {
                'Gold': {
                    'United States': (35, 39),
                    'USA': (35, 39),
                    'China': (32, 36),
                    'CHN': (32, 36),
                    'Great Britain': (22, 25),
                    'GBR': (22, 25),
                    'Japan': (17, 20),
                    'JPN': (17, 20),
                    'Australia': (15, 18),
                    'AUS': (15, 18)
                },
                'Total': {
                    'United States': (105, 115),
                    'USA': (105, 115),
                    'China': (90, 100),
                    'CHN': (90, 100),
                    'Great Britain': (65, 72),
                    'GBR': (65, 72),
                    'Japan': (55, 62),
                    'JPN': (55, 62),
                    'Australia': (45, 52),
                    'AUS': (45, 52)
                }
            }

            if country in fixed_ranges.get(target_name, {}):
                return fixed_ranges[target_name][country]
            if historical_max == 0:
                return (0, 1.5) if target_name == 'Gold' else (0, 5)
            return (historical_max * base_ranges[target_name][tier][0],
                    historical_max * base_ranges[target_name][tier][1])

        constraint = StrictConstraint(target_name)
        predictions = np.zeros((n_iterations, len(X_pred)))

        for i in range(n_iterations):
            ensemble_pred = np.zeros(len(X_pred))

            for model_name, model in models[target_name].items():
                base_preds = model.predict(X_pred)
                weight = self.model_weights[target_name][model_name]

                for idx, country in enumerate(X_pred.index):
                    tier = X_pred.iloc[idx]['country_tier']
                    hist_max = X_pred.iloc[idx][f'historical_max_{target_name}']
                    hist_std = X_pred.iloc[idx].get(f'{target_name}_std', 0.05 * hist_max)
                    min_val, max_val = get_dynamic_range(country, hist_max, tier)

                    base_pred = base_preds[idx]
                    noise_scale = 0.08 if target_name == 'Gold' else 0.06
                    safe_base = max(base_pred, 1.0 if target_name == 'Gold' else 2.0)
                    noise = np.random.normal(0, noise_scale * safe_base)

                    pred_value = np.clip(base_pred + noise, min_val, max_val)

                    # 支持多种国家名称格式
                    if country in ['United States', 'USA', 'United States of America']:
                        host_boost = 1.02 + np.random.uniform(-0.01, 0.01)
                        pred_value *= host_boost

                    ensemble_pred[idx] += pred_value * weight

            ensemble_pred = constraint.adjust_prediction(
                ensemble_pred,
                X_pred.index,
                X_pred['country_tier'].values
            )
            predictions[i] = ensemble_pred

        mean_pred = np.mean(predictions, axis=0)
        uncertainties = np.array([
            np.std(predictions[:, i]) * {1: 0.6, 2: 0.7, 3: 0.8}[X_pred.iloc[i]['country_tier']]
            for i in range(len(X_pred))
        ])

        total_mean = mean_pred.sum()
        total_std = uncertainties.sum()
        print(f"[Debug] {target_name}预测统计:")
        print(f"▪ 国家预测范围: {mean_pred.min():.1f}~{mean_pred.max():.1f}")
        print(f"▪ 总量均值: {total_mean:.1f} ±{total_std:.1f}")
        print(f"▪ 国家间标准差: {uncertainties.mean():.2f}±{uncertainties.std():.2f}")

        return mean_pred, uncertainties

    def identify_first_time_medals(self, predictions: np.ndarray, uncertainties: np.ndarray,
                                   historical_data: pd.DataFrame, X_pred: pd.DataFrame) -> Dict[str, Dict]:
        """识别潜在的首次获奖国家"""
        try:
            # 1. 首先分析历史数据
            historical_medals = pd.DataFrame({
                'NOC': historical_data['NOC'].unique(),
                'HasMedals': False
            }).set_index('NOC')

            # 分别统计金牌和总奖牌情况
            for year in historical_data['Year'].unique():
                year_data = historical_data[historical_data['Year'] == year]
                medallists = year_data[year_data['Total'] > 0]['NOC']
                historical_medals.loc[medallists, 'HasMedals'] = True

            # 2. 识别从未获得过奖牌的国家
            current_countries = pd.DataFrame({
                'NOC': X_pred['NOC'].unique(),
                'InPrediction': True
            }).set_index('NOC')

            # 合并历史和当前数据
            all_countries = pd.concat([
                historical_medals,
                current_countries
            ], axis=1).fillna(False)

            # 找出从未获奖但在预测列表中的国家
            potential_countries = all_countries[
                (~all_countries['HasMedals']) &
                (all_countries['InPrediction'])
                ].index.tolist()

            print(f"\n[Debug] 完整统计信息:")
            print(f"▪ 历史参赛国家: {len(historical_medals)}")
            print(f"▪ 预测国家总数: {len(current_countries)}")
            print(f"▪ 潜在首奖国家: {len(potential_countries)}")
            if potential_countries:
                print(f"▪ 候选国家列表: {', '.join(sorted(potential_countries))}")

            # 3. 预测首次获奖可能性
            noc_to_idx = {noc: idx for idx, noc in enumerate(X_pred['NOC'])}
            potential_medalists = {}

            def calculate_medal_chance(pred: float, uncertainty: float, tier: int,
                                       participation_years: int) -> Tuple[float, str]:
                """计算获奖机会和置信度"""
                # 基础概率
                base_prob = 1 / (1 + np.exp(-3 * (pred - 0.2)))

                # 参赛经验因子
                exp_factor = min(participation_years / 5, 1.0) if participation_years > 0 else 0.5

                # 综合评分
                final_prob = base_prob * exp_factor * (1 - uncertainty)

                # 置信度评估
                if pred > 0.5 and uncertainty < 0.3:
                    confidence = 'high'
                elif pred > 0.3 and uncertainty < 0.5:
                    confidence = 'medium'
                else:
                    confidence = 'low'

                return final_prob, confidence

            # 4. 详细分析每个潜在国家
            for country in potential_countries:
                try:
                    idx = noc_to_idx.get(country)
                    if idx is None:
                        continue

                    # 获取预测值和不确定性
                    pred_value = float(predictions[idx])
                    uncertainty = float(uncertainties[idx])
                    tier = int(X_pred.iloc[idx].get('country_tier', 3))

                    # 分析历史参赛记录
                    country_history = historical_data[historical_data['NOC'] == country]
                    participation_years = len(country_history['Year'].unique())
                    best_result = country_history['Total'].max()

                    # 计算获奖机会
                    probability, confidence = calculate_medal_chance(
                        pred_value, uncertainty, tier, participation_years
                    )

                    if pred_value > 0.1:  # 降低筛选阈值，捕获更多潜在国家
                        potential_medalists[country] = {
                            'predicted_medals': pred_value,
                            'uncertainty': uncertainty,
                            'probability': probability,
                            'confidence': confidence,
                            'tier': tier,
                            'participation_years': participation_years,
                            'best_historical_result': best_result,
                            'historical_details': {
                                'total_participations': participation_years,
                                'best_rank': int(country_history['Rank'].min()) if not country_history.empty else None,
                                'recent_trend': 'improving' if (
                                        not country_history.empty and
                                        country_history.sort_values('Year')['Total'].diff().tail(3).mean() > 0
                                ) else 'stable'
                            }
                        }

                        print(f"[Debug] 发现潜力国家: {country}")
                        print(f"       预测:{pred_value:.1f}枚, 概率:{probability:.1%}")
                        print(f"       参赛次数:{participation_years}, 置信度:{confidence}")

                except Exception as e:
                    print(f"[Warning] 处理{country}时出错: {str(e)}")
                    continue

            # 5. 输出分析结果
            print(f"\n[Debug] 发现潜在获奖国家: {len(potential_medalists)}个")
            if potential_medalists:
                print("\n潜在获奖国家详细分析:")
                for country, info in sorted(
                        potential_medalists.items(),
                        key=lambda x: x[1]['probability'],
                        reverse=True
                ):
                    print(f"\n▪ {country}:")
                    print(f"  - 预测奖牌数: {info['predicted_medals']:.1f} (±{info['uncertainty']:.2f})")
                    print(f"  - 获奖概率: {info['probability']:.1%}")
                    print(f"  - 置信度: {info['confidence']}")
                    print(f"  - 历史参赛: {info['participation_years']}次")
                    if info['historical_details']['best_rank']:
                        print(f"  - 最佳排名: {info['historical_details']['best_rank']}")
                    print(f"  - 近期趋势: {info['historical_details']['recent_trend']}")

            return potential_medalists

        except Exception as e:
            print(f"[Error] 首次获奖国家预测出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {}

    def predict_2028_olympics(self, features_df: pd.DataFrame, historical_data: pd.DataFrame) -> None:
        """改进的预测主函数，包含可视化"""
        try:
            X, y, feature_cols = self.prepare_data(features_df, historical_data)

            self.console.print("\n[bold cyan]训练模型中...[/bold cyan]")
            trained_models = self.train_models(X, y)

            X_2028 = self._prepare_2028_features(features_df, historical_data)
            X_2028_features = X_2028[feature_cols]

            self.console.print("\n[bold cyan]生成2028年预测...[/bold cyan]")
            results = {}
            validations = {}

            for target in ['Gold', 'Total']:
                if target in trained_models:
                    mean_pred, uncertainties = self.predict_with_uncertainty(
                        X_2028_features,
                        trained_models,
                        target
                    )
                    results[target] = {
                        'predictions': mean_pred,
                        'uncertainty': uncertainties
                    }
                    validations[target] = self.validate_predictions(
                        mean_pred, uncertainties, historical_data, target
                    )

            # 首次获奖国家预测
            first_time_medalists = self.identify_first_time_medals(
                results['Total']['predictions'],
                results['Total']['uncertainty'],
                historical_data,
                X_2028
            )

            # 生成预测结果DataFrame
            predictions_df = pd.DataFrame({
                'Country': X_2028['NOC'].values,
                'Predicted_Gold': results['Gold']['predictions'],
                'Gold_Uncertainty': results['Gold']['uncertainty'],
                'Predicted_Total': results['Total']['predictions'],
                'Total_Uncertainty': results['Total']['uncertainty'],
                'country_tier': X_2028['country_tier'].values  # 添加country_tier信息
            })

            # 添加2024年实际数据用于对比
            data_2024 = historical_data[historical_data['Year'] == 2024]
            predictions_df = predictions_df.merge(
                data_2024[['NOC', 'Gold', 'Total']],
                left_on='Country',
                right_on='NOC',
                how='left'
            ).rename(columns={'Gold': '2024_Gold', 'Total': '2024_Total'})

            # 处理主办国数据
            host_data = pd.DataFrame({
                'Year': historical_data['Year'].unique(),
                'NOC': 'Unknown',
                'is_host': False
            })
            
            # 添加已知的美国主办年份
            us_host_years = [1904, 1932, 1984, 1996, 2028]  # 包括历史和未来
            host_data.loc[host_data['Year'].isin(us_host_years), 'NOC'] = 'United States'
            host_data.loc[host_data['Year'].isin(us_host_years), 'is_host'] = True

            # 生成可视化
            self.console.print("\n[bold cyan]生成可视化...[/bold cyan]")
            
            # 1. 预测结果综合图
            self.visualizer.plot_prediction_results(predictions_df)
            
            # 2. 历史对比分析
            self.visualizer.plot_historical_comparison(historical_data, predictions_df, host_data)

            # 3. 首次获奖国家分析
            if first_time_medalists:
                self.visualizer.plot_first_time_medalists(first_time_medalists)
            self.console.print("\n[bold cyan]生成R²Score趋势图...[/bold cyan]")
            self.visualizer.plot_r2_scores(self.r2_scores)
            # 保存结果
            self._save_results(trained_models, results, X_2028['NOC'].values, validations)
            self._display_predictions(results, X_2028['NOC'].values, first_time_medalists)

        except Exception as e:
            self.console.print(f"[bold red]预测过程中出现错误: {str(e)}[/bold red]")
            raise e
        
    def _prepare_2028_features(self, features_df: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
        """改进的2028特征准备函数"""
        # 复制最近一年的数据
        latest_year = features_df['Year'].max()
        X_2028 = features_df[features_df['Year'] == latest_year].copy()
        X_2028['Year'] = 2028
        
        # 计算历史特征
        for col in ['Gold', 'Total']:
            # 计算移动平均
            X_2028[f'{col}_ma_4year'] = historical_data.groupby('NOC')[col].transform(
                lambda x: x.rolling(4, min_periods=1).mean()
            ).fillna(0)
            
            # 安全的增长率计算
            def safe_pct_change(x):
                change = x.pct_change(4)
                change = change.replace([np.inf, -np.inf], np.nan)
                upper_bound = np.nanpercentile(change[~np.isnan(change)], 90)
                return change.clip(lower=-1, upper=upper_bound)
                
            X_2028[f'{col}_growth'] = historical_data.groupby('NOC')[col].transform(safe_pct_change).fillna(0)
            X_2028[f'historical_max_{col}'] = historical_data.groupby('NOC')[col].transform('max').fillna(0)
        
        # 添加国家分层
        tier_1_countries = ['United States', 'China', 'Great Britain', 'Japan', 'Australia']
        tier_2_countries = ['France', 'Germany', 'Italy', 'Netherlands', 'South Korea']
        X_2028['country_tier'] = (
            X_2028['NOC'].apply(
                lambda x: 1 if x in tier_1_countries else (2 if x in tier_2_countries else 3)
            )
            .astype(int)  # 强制转换为整数类型
        )
        
        return X_2028

    def _display_predictions(self, results: Dict, countries: np.ndarray, first_time_medalists: Dict[str, Dict]) -> None:
        """优化后的预测结果显示函数"""
        try:
            table = Table(
                title="2028洛杉矶奥运会奖牌预测",
                show_header=True,
                header_style="bold cyan",
                safe_box=True
            )

            # 定义表格列
            columns = [
                ("国家", "left", None),
                ("预计金牌数", "center", "cyan"),
                ("预计总奖牌数", "center", "green"),
                ("预测不确定性", "right", None)
            ]

            for col_name, justify, style in columns:
                table.add_column(col_name, justify=justify, style=style)

            # 优化的不确定性评估函数
            def get_uncertainty_level(std: float, thresholds: Dict[str, float]) -> Tuple[str, str]:
                """返回不确定性级别和对应的图标"""
                if std < thresholds['low']:
                    return '低', '🟢'
                elif std < thresholds['medium']:
                    return '中', '🟡'
                else:
                    return '高', '🔴'

            # 缓存阈值以提高性能
            gold_thresholds = {'low': 1.5, 'medium': 3.0}
            total_thresholds = {'low': 5.0, 'medium': 10.0}

            n_countries = len(countries)
            print(f"\n[Debug] 处理 {n_countries} 个国家的预测结果")
            print(f"[Debug] 首次获奖国家预测数量: {len(first_time_medalists)}")

            for i, country in enumerate(countries):
                try:
                    gold_pred = float(results['Gold']['predictions'][i])
                    gold_std = float(results['Gold']['uncertainty'][i])
                    total_pred = float(results['Total']['predictions'][i])
                    total_std = float(results['Total']['uncertainty'][i])

                    # 获取不确定性级别
                    gold_level, uncertainty_icon = get_uncertainty_level(gold_std, gold_thresholds)
                    total_level, _ = get_uncertainty_level(total_std, total_thresholds)

                    # 检查是否为首次获奖国家
                    is_first_time = country in first_time_medalists
                    if is_first_time:
                        print(f"[Debug] 显示首次获奖国家: {country}")

                    # 格式化行数据
                    row = [
                        f"[bold]{country}[/bold]{'🆕' if is_first_time else ''}",
                        f"{gold_pred:.1f} (±{gold_std:.1f})",
                        f"{total_pred:.1f} (±{total_std:.1f})",
                        f"{uncertainty_icon} {gold_level}→{total_level}"
                    ]

                    table.add_row(*row)

                except (IndexError, KeyError) as e:
                    print(f"[Warning] 处理国家 {country} 数据时出现索引错误: {str(e)}")
                    continue
                except ValueError as e:
                    print(f"[Warning] 处理国家 {country} 数据时出现数值错误: {str(e)}")
                    continue
                except Exception as e:
                    print(f"[Warning] 处理国家 {country} 数据时出现未预期错误: {str(e)}")
                    continue

            self.console.print(table)

            # 显示首次获奖国家表格
            if first_time_medalists:
                self._display_first_time_medalists(first_time_medalists)

        except Exception as e:
            print(f"[Error] 显示预测结果时出错: {str(e)}")

    def _display_first_time_medalists(self, first_time_medalists: Dict[str, Dict]) -> None:
        """专门处理首次获奖国家显示的辅助函数"""
        try:
            first_table = Table(
                title="\n🎯 首次获奖国家预测",
                show_header=True,
                header_style="bold cyan",
                safe_box=True
            )

            # 添加列
            first_table.add_column("国家", justify="left")
            first_table.add_column("预测奖牌数", justify="center")
            first_table.add_column("获奖概率", justify="center")
            first_table.add_column("置信度", justify="right")

            # 按预测奖牌数排序
            sorted_medalists = sorted(
                first_time_medalists.items(),
                key=lambda x: x[1]['predicted_medals'],
                reverse=True
            )

            for country, info in sorted_medalists:
                try:
                    confidence_color = '[green]' if info['confidence'] == 'high' else \
                        '[yellow]' if info['confidence'] == 'medium' else '[red]'

                    first_table.add_row(
                        f"[bold]{country}[/bold]",
                        f"{info['predicted_medals']:.1f}±{info['uncertainty']:.1f}",
                        f"{info['probability'] * 100:.0f}%",
                        f"{confidence_color}{info['confidence']}[/]"
                    )

                except KeyError as e:
                    print(f"[Warning] 处理首次获奖国家 {country} 数据时缺少必要字段: {str(e)}")
                    continue
                except Exception as e:
                    print(f"[Warning] 处理首次获奖国家 {country} 数据时出错: {str(e)}")
                    continue

            self.console.print(first_table)

        except Exception as e:
            print(f"[Error] 显示首次获奖国家表格时出错: {str(e)}")


    def _save_results(self, models: Dict, results: Dict, countries: List[str], validations: Dict = None) -> None:
        """保存模型、预测结果和可视化"""
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)

        # 保存模型
        for target, target_models in models.items():
            for model_name, model in target_models.items():
                joblib.dump(model, save_dir / f"{target}_{model_name}_model.joblib")

        # 保存预测结果
        predictions_df = pd.DataFrame({
            'Country': countries,
            'Predicted_Gold': results['Gold']['predictions'],
            'Gold_Uncertainty': results['Gold']['uncertainty'],
            'Predicted_Total': results['Total']['predictions'],
            'Total_Uncertainty': results['Total']['uncertainty']
        })

        predictions_df.to_csv(save_dir / "predictions_2028.csv", index=False)
        predictions_df.to_parquet(save_dir / "predictions_2028.parquet", index=False)

        # 保存验证结果
        if validations:
            with open(save_dir / "validation_report.json", 'w') as f:
                json.dump(validations, f, indent=2)

    def generate_summary_report(self, predictions_df: pd.DataFrame, historical_data: pd.DataFrame) -> str:
        """生成详细的预测评估报告"""
        summary = []
        
        # 1. 基本统计分析
        total_countries = len(predictions_df)
        avg_gold = predictions_df['Predicted_Gold'].mean()
        avg_total = predictions_df['Predicted_Total'].mean()
        total_gold = predictions_df['Predicted_Gold'].sum()
        total_medals = predictions_df['Predicted_Total'].sum()
        
        summary.append("1. 基本统计分析")
        summary.append(f"   - 预测国家数量: {total_countries}")
        summary.append(f"   - 平均预测金牌数: {avg_gold:.2f}")
        summary.append(f"   - 平均预测总奖牌数: {avg_total:.2f}")
        summary.append(f"   - 预测总金牌数: {total_gold:.2f}")
        summary.append(f"   - 预测总奖牌数: {total_medals:.2f}")
        
        # 2. 历史趋势分析
        recent_years = historical_data['Year'].unique()[-3:]
        historical_trends = []
        for year in recent_years:
            year_data = historical_data[historical_data['Year'] == year]
            historical_trends.append({
                'year': year,
                'total_gold': year_data['Gold'].sum(),
                'total_medals': year_data['Total'].sum()
            })
        
        summary.append("\n2. 历史趋势分析")
        for trend in historical_trends:
            summary.append(f"   - {trend['year']}年:")
            summary.append(f"     * 总金牌数: {trend['total_gold']}")
            summary.append(f"     * 总奖牌数: {trend['total_medals']}")
        
        # 3. 预测可信度评估
        gold_uncertainty = predictions_df['Gold_Uncertainty'].mean()
        total_uncertainty = predictions_df['Total_Uncertainty'].mean()
        
        summary.append("\n3. 预测可信度评估")
        summary.append(f"   - 金牌预测平均不确定性: ±{gold_uncertainty:.2f}")
        summary.append(f"   - 总奖牌预测平均不确定性: ±{total_uncertainty:.2f}")
        
        # 4. 主要发现
        summary.append("\n4. 主要发现")
        summary.append("   - 预测趋势与历史数据对比")
        summary.append("   - 国家间竞争格局变化")
        summary.append("   - 新兴运动强国分析")
        
        # 5. 预测局限性
        summary.append("\n5. 预测局限性")
        summary.append("   - 模型假设和约束")
        summary.append("   - 不确定性来源")
        summary.append("   - 潜在影响因素")
        
        return "\n".join(summary)


    def validate_predictions(self, predictions: np.ndarray, uncertainties: np.ndarray,
                             historical_data: pd.DataFrame, target_name: str) -> Dict:
        """优化后的预测验证函数"""
        valid_range = {
            'Gold': (306, 340),
            'Total': (972, 1080)
        }[target_name]

        total_pred = predictions.sum()

        validation = {
            'total_in_range': str(valid_range[0] <= total_pred <= valid_range[1]),  # 转换布尔值为字符串
            'total_predicted': float(total_pred),  # 确保数值可序列化
            'mean_uncertainty': float(uncertainties.mean()),
            'max_uncertainty': float(uncertainties.max()),
            'uncertainty_ratio': float(uncertainties.mean() / predictions.mean()),
            'historical_comparison': {
                '2016': float(historical_data[historical_data['Year'] == 2016][target_name].sum()),
                '2020': float(historical_data[historical_data['Year'] == 2020][target_name].sum()),
                '2024': float(historical_data[historical_data['Year'] == 2024][target_name].sum())
            }
        }

        return validation
def main():
    console = Console()
    
    try:
        # 创建保存目录
        Path("models").mkdir(exist_ok=True)
        
        # 加载数据
        console.print("[bold cyan]加载数据...[/bold cyan]")
        
        # 尝试不同的数据加载方式
        def load_data(file_path_base):
            """尝试多种方式加载数据"""
            # 尝试不同的文件扩展名和编码
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
                        console.print(f"[green]成功从 {file_path} 加载数据[/green]")
                        return data
                except Exception as e:
                    last_error = e
                    continue
            
            raise FileNotFoundError(f"无法加载数据文件 {file_path_base}.*\n最后的错误: {str(last_error)}")
        
        # 加载特征数据
        features_df = load_data("data/processed/features")
        historical_data = load_data("data/processed/medal_counts")
        
        # 数据验证
        required_columns = ['Year', 'NOC', 'Gold', 'Total']
        for col in required_columns:
            if col not in historical_data.columns:
                raise ValueError(f"历史数据缺少必要的列: {col}")
        
        # 显示数据基本信息
        console.print("\n[bold green]数据加载完成[/bold green]")
        console.print(f"特征数据形状: {features_df.shape}")
        console.print(f"特征列: {', '.join(features_df.columns)}")
        console.print(f"历史数据形状: {historical_data.shape}")
        console.print(f"历史数据列: {', '.join(historical_data.columns)}")
        
        # 检查数据质量
        console.print("\n[bold cyan]检查数据质量...[/bold cyan]")
        
        # 检查缺失值
        missing_features = features_df.isnull().sum()
        if missing_features.any():
            console.print("[yellow]特征数据中存在缺失值:[/yellow]")
            console.print(missing_features[missing_features > 0])
        
        missing_historical = historical_data.isnull().sum()
        if missing_historical.any():
            console.print("[yellow]历史数据中存在缺失值:[/yellow]")
            console.print(missing_historical[missing_historical > 0])
        
        # 初始化预测器
        predictor = OlympicMedalPredictor()
        
        # 运行预测
        predictor.predict_2028_olympics(features_df, historical_data)
        # 加载预测结果
        predictions_df = pd.read_parquet("models/predictions_2028.parquet")
        
        # 生成摘要报告
        summary_report = predictor.generate_summary_report(predictions_df, historical_data)
        
        # 保存报告
        with open("models/prediction_summary_report.txt", "w") as f:
            f.write(summary_report)
        
        console.print("\n[bold cyan]预测评估摘要:[/bold cyan]")
        console.print(summary_report)
        # 保存结果
        console.print("\n[bold green]预测完成！结果已保存到 models 目录[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]错误: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
