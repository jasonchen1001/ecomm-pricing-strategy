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
        
        # 拟合模型: ln(Q) = α + β*ln(P)
        self.model.fit(X, y)
        
        # 价格弹性系数是斜率的负值: ε = -β
        self.elasticity = -self.model.coef_[0]
        return self.elasticity
    
    def _point_elasticity(self, prices, quantities, segments=5):
        """点弹性方法"""
        # 按价格排序
        sorted_indices = np.argsort(prices)
        prices = prices[sorted_indices]
        quantities = quantities[sorted_indices]
        
        # 计算每个点的弹性
        elasticities = []
        for i in range(1, len(prices)):
            p1, p2 = prices[i-1], prices[i]
            q1, q2 = quantities[i-1], quantities[i]
            
            # 避免除以零和极小值
            if p1 < 1e-6 or p2 < 1e-6 or q1 < 1e-6 or q2 < 1e-6:
                continue
            
            # 计算点弹性: (ΔQ/Q)/(ΔP/P)
            price_change = (p2 - p1) / ((p1 + p2) / 2)  # 使用中点公式
            quantity_change = (q2 - q1) / ((q1 + q2) / 2)
            
            if abs(price_change) > 1e-6:  # 避免除以接近零的值
                point_elasticity = quantity_change / price_change
                # 只保留合理范围内的弹性值
                if abs(point_elasticity) < 10:  # 设置合理的上限
                    elasticities.append(point_elasticity)
        
        # 返回平均弹性
        if elasticities:
            # 使用中位数而不是平均值，避免极端值的影响
            return np.median(elasticities)
        return 0
    
    def _arc_elasticity(self, prices, quantities, segments=5):
        """弧弹性方法"""
        # 计算相邻点之间的弧弹性
        arc_elasticities = []
        
        # 按价格排序
        sorted_indices = np.argsort(prices)
        prices = prices[sorted_indices]
        quantities = quantities[sorted_indices]
        
        # 将数据分成segments段并计算每段的平均值
        price_segments = np.array_split(prices, segments)
        quantity_segments = np.array_split(quantities, segments)
        
        avg_prices = [np.mean(p) for p in price_segments]
        avg_quantities = [np.mean(q) for q in quantity_segments]
        
        # 计算相邻段之间的弧弹性
        for i in range(len(avg_prices)-1):
            p1, p2 = avg_prices[i], avg_prices[i+1]
            q1, q2 = avg_quantities[i], avg_quantities[i+1]
            
            # 避免除以零和极小值
            if min(p1, p2, q1, q2) < 1e-6:
                continue
            
            # 计算弧弹性: ((Q2-Q1)/((Q1+Q2)/2))/((P2-P1)/((P1+P2)/2))
            dq = (q2 - q1)
            dp = (p2 - p1)
            q_avg = (q1 + q2) / 2
            p_avg = (p1 + p2) / 2
            
            if abs(dp/p_avg) > 1e-6:  # 避免除以接近零的值
                arc_elasticity = (dq/q_avg)/(dp/p_avg)
                if abs(arc_elasticity) < 10:  # 设置合理的上限
                    arc_elasticities.append(arc_elasticity)
        
        # 返回中位数弹性
        if arc_elasticities:
            return np.median(arc_elasticities)
        return 0
    
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