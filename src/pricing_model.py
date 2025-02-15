import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class PricingModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """准备模型特征"""
        features = pd.DataFrame()
        
        # 基础特征
        features['rating_count'] = np.log1p(df['rating_count'])  # 评论数（作为销量代理）
        
        # 清理折扣率数据（移除%符号并转换为浮点数）
        features['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float) / 100
        
        features['sentiment_score'] = df['sentiment_score']  # 评论情感得分
        features['rating'] = df['rating']  # 评分
        
        # 计算每个类别的平均价格
        category_avg_price = df.groupby('main_category')['discounted_price'].transform('mean')
        features['price_to_category_avg'] = df['discounted_price'] / category_avg_price
        
        # 计算综合得分
        features['composite_score'] = (
            0.4 * np.log1p(df['rating_count']) / np.log1p(df['rating_count'].max()) +  # 销量权重
            0.3 * df['rating'] / 5.0 +  # 评分权重
            0.3 * df['sentiment_score']  # 情感权重
        )
        
        # 标准化特征
        features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        return features
    
    def train(self, df):
        """训练定价模型"""
        print("\n=== Training Pricing Model ===")
        features = self.prepare_features(df)
        current_prices = df['discounted_price']
        
        # 训练模型
        self.model.fit(features, current_prices)
        
        # 计算特征重要性
        importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        for _, row in importance.iterrows():
            print(f"- {row['feature']}: {row['importance']:.3f}")
        
        return importance
    
    def recommend_prices(self, df):
        """生成价格建议"""
        print("\n=== Generating Price Recommendations ===")
        features = self.prepare_features(df)
        predicted_prices = self.model.predict(features)
        
        # 创建建议数据框
        recommendations = pd.DataFrame()
        recommendations['product_id'] = df['product_id']
        recommendations['current_price'] = df['discounted_price']
        recommendations['predicted_price'] = predicted_prices
        
        # 基于综合得分和情感分数调整价格
        # 降低基础调整幅度
        base_adjustment = (features['composite_score'] * 3)  # 从5降到3
        sentiment_adjustment = (df['sentiment_score'] - 0.5) * 2  # 从3降到2
        
        # 添加随机波动（±0.5%）
        np.random.seed(42)
        random_adjustment = np.random.uniform(-0.5, 0.5, len(recommendations))
        
        # 计算最终调整幅度
        recommendations['adjusted_change'] = (base_adjustment + sentiment_adjustment + random_adjustment)
        
        # 更保守的价格变动范围
        recommendations['adjusted_change'] = recommendations['adjusted_change'].clip(-5, 5)  # 最大变动±5%
        
        # 计算建议价格
        recommendations['recommended_price'] = recommendations['current_price'] * (
            1 + recommendations['adjusted_change'] / 100
        )
        
        # 添加置信度分数
        recommendations['confidence'] = self._calculate_confidence(df, features)
        
        # 生成建议
        recommendations['recommendation'] = recommendations.apply(
            self._get_recommendation, axis=1
        )
        
        # 计算统计信息
        total = len(recommendations)
        increase_mask = recommendations['adjusted_change'] > 0
        decrease_mask = recommendations['adjusted_change'] < 0
        no_change_mask = abs(recommendations['adjusted_change']) < 3
        
        increase_count = sum(increase_mask)
        decrease_count = sum(decrease_mask)
        no_change_count = sum(no_change_mask)
        
        # 计算收入变化
        current_revenue = (recommendations['current_price'] * df['rating_count']).sum()
        expected_revenue = (recommendations['recommended_price'] * df['rating_count']).sum()
        revenue_change = ((expected_revenue - current_revenue) / current_revenue * 100)
        
        # 计算每个产品的收入变化
        recommendations['current_revenue'] = recommendations['current_price'] * df['rating_count']
        recommendations['expected_revenue'] = recommendations['recommended_price'] * df['rating_count']
        recommendations['revenue_change_pct'] = (
            (recommendations['expected_revenue'] - recommendations['current_revenue']) 
            / recommendations['current_revenue'] * 100
        )
        
        # 输出更详细的统计信息
        print("\n=== Price Adjustment Statistics ===")
        print(f"Total products analyzed: {total}")
        
        # 更细致的价格变动区间统计
        change_ranges = {
            '降价(3-5%)': (-5, -3),
            '小幅降价(1-3%)': (-3, -1),
            '基本维持(±1%)': (-1, 1),
            '小幅提价(1-3%)': (1, 3),
            '提价(3-5%)': (3, 5)
        }
        
        print("\nPrice Change Distribution:")
        for name, (lower, upper) in change_ranges.items():
            count = sum((recommendations['adjusted_change'] >= lower) & 
                       (recommendations['adjusted_change'] < upper))
            print(f"{name}: {count} products ({count/total*100:.1f}%)")
        
        print(f"\nRevenue Impact:")
        print(f"Current total revenue: ₹{current_revenue:,.2f}")
        print(f"Expected total revenue: ₹{expected_revenue:,.2f}")
        print(f"Expected revenue change: {revenue_change:.1f}%")
        
        # 按调价幅度排序显示top变动
        print("\nTop Price Increases:")
        top_increases = recommendations.nlargest(5, 'adjusted_change')
        for _, row in top_increases.iterrows():
            print(f"Product {row['product_id']}: +{row['adjusted_change']:.1f}% (₹{row['current_price']:.2f} → ₹{row['recommended_price']:.2f})")
        
        print("\nTop Price Decreases:")
        top_decreases = recommendations.nsmallest(5, 'adjusted_change')
        for _, row in top_decreases.iterrows():
            print(f"Product {row['product_id']}: {row['adjusted_change']:.1f}% (₹{row['current_price']:.2f} → ₹{row['recommended_price']:.2f})")
        
        return recommendations
    
    def _calculate_confidence(self, df, features):
        """计算建议的置信度"""
        confidence = pd.Series(index=df.index)
        
        # 基于评论数的置信度
        review_confidence = np.clip(df['rating_count'] / df['rating_count'].quantile(0.9), 0, 1)
        
        # 基于情感分数的置信度（越极端越确信）
        sentiment_confidence = abs(df['sentiment_score'] - 0.5) * 2
        
        # 基于价格偏离度的置信度
        price_deviation = abs(features['price_to_category_avg'])
        price_confidence = 1 - np.clip(price_deviation, 0, 1)
        
        # 综合置信度
        confidence = (
            review_confidence * 0.4 +
            sentiment_confidence * 0.3 +
            price_confidence * 0.3
        )
        
        return confidence
    
    def _get_recommendation(self, row):
        """生成具体的价格调整建议"""
        change = row['adjusted_change']
        confidence = row['confidence']
        
        if confidence < 0.3:
            return "数据不足，建议观察"
        
        if abs(change) < 3:
            return "价格合理，保持现状"
        elif change > 0:
            return f"建议提价 {change:.1f}%"
        else:
            return f"建议降价 {abs(change):.1f}%"

def main():
    """测试定价模型"""
    try:
        # 加载数据
        print("=== Loading Data ===")
        df = pd.read_csv('data/processed_amazon.csv')
        
        # 创建并训练模型
        model = PricingModel()
        importance = model.train(df)
        
        # 生成价格建议
        recommendations = model.recommend_prices(df)
        
        # 显示部分结果
        print("\n=== Sample Recommendations ===")
        sample = recommendations.head()
        for _, row in sample.iterrows():
            print(f"\nProduct ID: {row['product_id']}")
            print(f"Current Price: ₹{row['current_price']:.2f}")
            print(f"Recommended Price: ₹{row['recommended_price']:.2f}")
            print(f"Change: {row['adjusted_change']:.1f}%")
            print(f"Confidence: {row['confidence']:.2f}")
            print(f"Recommendation: {row['recommendation']}")
        
        # 保存建议
        recommendations.to_csv('data/price_recommendations.csv', index=False)
        print("\nRecommendations saved to price_recommendations.csv")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 