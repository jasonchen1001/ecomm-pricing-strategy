from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DynamicPricingModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, features, target):
        """训练模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        return score
    
    def predict_prices(self, features):
        """预测最优价格"""
        predictions = self.model.predict(features)
        return predictions
    
    def generate_recommendations(self, df, predictions):
        """生成价格建议"""
        recommendations = pd.DataFrame({
            'product_id': df['product_id'],
            'current_price': df['discounted_price'],
            'recommended_price': predictions,
            'expected_change': ((predictions - df['discounted_price'])/df['discounted_price']*100)
        })
        return recommendations.sort_values('expected_change', ascending=False) 