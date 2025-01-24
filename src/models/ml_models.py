from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class GBMModel(BaseModel):
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y):
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        # 确保特征列顺序一致
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)