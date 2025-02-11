import pandas as pd
from pathlib import Path
from data_preprocessing import load_data, extract_features
from price_elasticity import PriceElasticityAnalyzer
from sentiment_analysis import SentimentAnalyzer
from pricing_model import DynamicPricingModel

def main():
    # 创建输出目录
    Path('outputs').mkdir(exist_ok=True)
    
    # 1. 数据预处理
    print("正在加载数据...")
    df = load_data('amazon.csv')
    features = extract_features(df)
    
    # 2. 价格弹性分析
    print("正在分析价格弹性...")
    elasticity_analyzer = PriceElasticityAnalyzer()
    elasticity = elasticity_analyzer.calculate_elasticity(
        df['discounted_price'].values,
        df['rating_count'].values
    )
    elasticity_analyzer.plot_elasticity(
        df['discounted_price'].values,
        df['rating_count'].values
    )
    
    # 3. 情感分析
    print("正在进行情感分析...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_reviews(df)
    sentiment_analyzer.generate_word_cloud(df)
    
    # 4. 动态定价模型
    print("正在构建定价模型...")
    pricing_model = DynamicPricingModel()
    model_score = pricing_model.train(
        features.drop(['price_segment', 'popularity'], axis=1),
        df['discounted_price']
    )
    
    predictions = pricing_model.predict_prices(
        features.drop(['price_segment', 'popularity'], axis=1)
    )
    
    recommendations = pricing_model.generate_recommendations(df, predictions)
    
    # 5. 生成报告
    generate_report(df, recommendations, elasticity, model_score)
    
def generate_report(df, recommendations, elasticity, model_score):
    """生成分析报告"""
    report = f"""
    印度电商线缆产品定价策略分析报告
    
    1. 市场概况
    ===========
    - 产品总数: {len(df)}
    - 平均折扣率: {df['real_discount'].mean():.2f}%
    - 平均评分: {df['rating'].mean():.2f}
    - 价格弹性系数: {elasticity:.2f}
    
    2. 情感分析
    ===========
    - 正面评价占比: {(df['sentiment'] > 0).mean()*100:.2f}%
    - 负面评价占比: {(df['sentiment'] < 0).mean()*100:.2f}%
    
    3. 定价模型表现
    ==============
    - 模型准确率: {model_score:.2f}
    
    4. 价格优化建议
    ==============
    - 建议提价产品数: {len(recommendations[recommendations['expected_change'] > 0])}
    - 建议降价产品数: {len(recommendations[recommendations['expected_change'] < 0])}
    - 预期平均利润提升: {recommendations['expected_change'].mean():.2f}%
    
    5. 重点关注产品TOP5
    ==================
    {recommendations.head().to_string()}
    """
    
    with open('outputs/report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("报告已生成到 outputs/report.txt")

if __name__ == "__main__":
    main() 