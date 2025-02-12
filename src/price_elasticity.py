import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PriceElasticityAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        self.elasticity = None
    
    def calculate_elasticity(self, prices, quantities, method='Log-Log', segments=5):
        """计算价格弹性"""
        if method == 'Log-Log':
            return self._log_log_elasticity(prices, quantities)
        elif method == 'Point':
            return self._point_elasticity(prices, quantities, segments)
        elif method == 'Arc':
            return self._arc_elasticity(prices, quantities, segments)
    
    def _log_log_elasticity(self, prices, quantities):
        """对数回归方法"""
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
    
    def _point_elasticity(self, prices, quantities, segments):
        """点弹性方法"""
        # 将数据转换为pandas Series
        prices_series = pd.Series(prices)
        quantities_series = pd.Series(quantities)
        
        # 将价格分成segments个区间
        price_bins = pd.qcut(prices_series, segments)
        avg_quantities = quantities_series.groupby(price_bins).mean()
        avg_prices = prices_series.groupby(price_bins).mean()
        
        # 计算相邻区间的弹性
        elasticities = []
        for i in range(len(avg_prices)-1):
            p1, p2 = avg_prices.iloc[i:i+2]
            q1, q2 = avg_quantities.iloc[i:i+2]
            e = ((q2-q1)/q1) / ((p2-p1)/p1)
            elasticities.append(e)
        
        self.elasticity = np.mean(elasticities)
        return self.elasticity
    
    def _arc_elasticity(self, prices, quantities, segments):
        """弧弹性方法"""
        # 将数据转换为pandas Series
        prices_series = pd.Series(prices)
        quantities_series = pd.Series(quantities)
        
        # 将价格分成segments个区间
        price_bins = pd.qcut(prices_series, segments)
        avg_quantities = quantities_series.groupby(price_bins).mean()
        avg_prices = prices_series.groupby(price_bins).mean()
        
        elasticities = []
        for i in range(len(avg_prices)-1):
            p1, p2 = avg_prices.iloc[i:i+2]
            q1, q2 = avg_quantities.iloc[i:i+2]
            e = ((q2-q1)/(q2+q1)) / ((p2-p1)/(p2+p1))
            elasticities.append(e)
        
        self.elasticity = np.mean(elasticities)
        return self.elasticity
    
    def plot_elasticity(self, prices, quantities, method='Log-Log', segments=5):
        """绘制价格-需求关系图"""
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 基础散点图
        plt.scatter(prices, quantities, alpha=0.3, color='gray', label='Raw Data')
        
        if method == 'Log-Log':
            # 对数回归拟合线
            mask = (prices > 0) & (quantities > 0)
            prices_clean = prices[mask]
            quantities_clean = quantities[mask]
            
            X = np.log(prices_clean)
            y = np.log(quantities_clean)
            
            # 预测值
            X_range = np.linspace(X.min(), X.max(), 100)
            y_pred = self.model.predict(X_range.reshape(-1, 1))
            
            # 转换回原始尺度
            plt.plot(np.exp(X_range), np.exp(y_pred), 'r-', 
                    label=f'Log-Log Fit (e={self.elasticity:.2f})')
            
        elif method in ['Point', 'Arc']:
            # 将数据转换为pandas Series
            prices_series = pd.Series(prices)
            quantities_series = pd.Series(quantities)
            
            # 计算分段平均值
            price_bins = pd.qcut(prices_series, segments)
            avg_quantities = quantities_series.groupby(price_bins).mean()
            avg_prices = prices_series.groupby(price_bins).mean()
            
            # 绘制分段点和连线
            plt.plot(avg_prices, avg_quantities, 'ro-', 
                    label=f'{method} Elasticity (e={self.elasticity:.2f})')
            plt.scatter(avg_prices, avg_quantities, color='red', s=100)
            
            # 添加区间标记
            for i, (p, q) in enumerate(zip(avg_prices, avg_quantities)):
                plt.annotate(f'Segment {i+1}', 
                           (p, q), 
                           xytext=(10, 10),
                           textcoords='offset points')
        
        plt.xlabel('Price (₹)')
        plt.ylabel('Demand (Reviews)')
        plt.title(f'Price-Demand Relationship ({method} Method)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加弹性说明
        if self.elasticity < 0.5:
            plt.figtext(0.02, 0.02, 
                       'Low Price Elasticity: Consumers are less sensitive to price changes',
                       color='green')
        else:
            plt.figtext(0.02, 0.02, 
                       'High Price Elasticity: Consumers are more sensitive to price changes',
                       color='red')
        
        plt.tight_layout()
        plt.savefig('outputs/price_elasticity.png', dpi=300, bbox_inches='tight')
        plt.close()

    def interpret_elasticity(self, threshold=0.5):
        """解释价格弹性系数"""
        # 确保有弹性系数值
        if not hasattr(self, 'elasticity') or self.elasticity is None:
            return """
            无法解释价格弹性：未计算弹性系数。
            请先调用 calculate_elasticity() 方法。
            """
        
        if self.elasticity < threshold:
            return f"""
            建议策略 (弹性系数 {self.elasticity:.2f} < {threshold:.2f}):
            1. 可以适当提高价格
            2. 重点关注产品质量和品牌建设
            3. 通过差异化竞争而非价格战
            """
        else:
            return f"""
            建议策略 (弹性系数 {self.elasticity:.2f} ≥ {threshold:.2f}):
            1. 保持价格竞争力
            2. 关注竞品定价
            3. 考虑促销策略
            """ 