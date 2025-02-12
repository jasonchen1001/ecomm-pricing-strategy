import pandas as pd
from pathlib import Path
from data_preprocessing import load_data, extract_features
from price_elasticity import PriceElasticityAnalyzer
from sentiment_analysis import SentimentAnalyzer
from pricing_model import DynamicPricingModel

def main():
    # 创建输出目录
    
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
    
    # 3. 情感分析
    print("正在进行情感分析...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_reviews(df)
    
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
    # 创建弹性分析器实例并设置弹性值
    elasticity_analyzer = PriceElasticityAnalyzer()
    elasticity_analyzer.elasticity = elasticity
    
    report = f"""# 印度电商线缆产品定价策略分析报告

## 1. 市场概况 📊
- **产品总数**: {len(df):,} 个
- **平均折扣率**: {df['real_discount'].mean():.1f}%
- **平均评分**: {df['rating'].mean():.2f} ⭐
- **价格弹性系数**: {elasticity:.2f}

## 2. 情感分析 💭
- **正面评价占比**: {(df['sentiment'] > 0).mean()*100:.1f}%
- **负面评价占比**: {(df['sentiment'] < 0).mean()*100:.1f}%

## 3. 定价模型表现 🎯
- **模型准确率**: {model_score:.2%}

## 4. 价格优化建议 💡
- **建议提价产品数**: {len(recommendations[recommendations['expected_change'] > 0]):,} 个
- **建议降价产品数**: {len(recommendations[recommendations['expected_change'] < 0]):,} 个
- **预期平均利润提升**: {recommendations['expected_change'].mean():.2f}%

## 5. 重点关注产品 TOP5 ⭐
| 产品ID | 当前价格 (₹) | 建议价格 (₹) | 预期变化 (%) |
|--------|-------------|--------------|--------------|
{recommendations.head().to_markdown(index=False)}

## 6. 策略建议 📈

### 价格弹性分析
{elasticity_analyzer.interpret_elasticity()}

### 市场定位建议
1. **高端市场**: 
   - 重点关注产品质量和品牌建设
   - 强调产品差异化
   - 维持较高利润率

2. **中端市场**:
   - 平衡价格和质量
   - 关注竞品定价
   - 保持稳定市场份额

3. **低端市场**:
   - 优化成本结构
   - 提高运营效率
   - 通过规模效应获利

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("报告已生成到 report.md")

if __name__ == "__main__":
    main() 