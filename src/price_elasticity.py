import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class PriceElasticityAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        self.elasticity = None
    
    def calculate_elasticity(self, prices, quantities):
        """计算价格弹性"""
        # 去除零值和负值
        mask = (prices > 0) & (quantities > 0)
        prices = prices[mask]
        quantities = quantities[mask]
        
        # 对数转换
        X = np.log(prices.reshape(-1, 1))
        y = np.log(quantities)
        
        # 去除NaN值
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # 拟合模型
        self.model.fit(X, y)
        
        # 价格弹性系数是斜率的负值
        self.elasticity = -self.model.coef_[0]
        return self.elasticity
    
    def plot_elasticity(self, prices, quantities):
        """绘制价格-需求关系图"""
        # 去除零值、负值和NaN
        mask = (prices > 0) & (quantities > 0)
        prices = prices[mask]
        quantities = quantities[mask]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(prices, quantities, alpha=0.5)
        plt.xlabel('Price (₹)')
        plt.ylabel('Demand (Reviews)')
        plt.title(f'Price Elasticity: {self.elasticity:.2f}')
        plt.savefig('outputs/price_elasticity.png')
        plt.close() 