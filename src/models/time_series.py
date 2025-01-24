from prophet import Prophet
import pandas as pd
from .base import BaseModel

class ProphetModel(BaseModel):
    def __init__(self):
        self.model = Prophet(yearly_seasonality=True)
        
    def fit(self, df):
        df = df.rename(columns={'Year': 'ds', 'Total': 'y'})
        self.model.fit(df)
        
    def predict(self, future_dates):
        future = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]