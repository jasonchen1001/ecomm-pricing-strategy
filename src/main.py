import pandas as pd
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
    print("\n=== 开始情感分析 ===")
    print("正在初始化情感分析器...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # 为了测试，先只分析前5条评论
    print("\n测试前5条评论的情感分析：")
    test_df = df.head()
    test_df = sentiment_analyzer.analyze_reviews(test_df)
    
    # 如果测试成功，再分析全部评论
    print("\n开始分析所有评论...")
    df = sentiment_analyzer.analyze_reviews(df)
    print("=== 情感分析完成 ===\n")
    
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
    
    # 计算情感分布
    positive_ratio = (df['sentiment'] > 0).mean() * 100
    negative_ratio = (df['sentiment'] < 0).mean() * 100
    neutral_ratio = (df['sentiment'] == 0).mean() * 100
    
    # 准备TOP5产品数据
    top5_products = recommendations.head()[['product_id', 'current_price', 'recommended_price', 'expected_change']]
    
    # 格式化数值列，确保输出格式正确
    top5_products = top5_products.round({
        'current_price': 2,
        'recommended_price': 2,
        'expected_change': 2
    })
    
    # 生成表格，不包含列名行
    table_rows = []
    for _, row in top5_products.iterrows():
        table_rows.append(f"| {row['product_id']} | {row['current_price']:.2f} | {row['recommended_price']:.2f} | {row['expected_change']:.2f} |")
    table_content = '\n'.join(table_rows)
    
    report = f"""# 印度电商线缆产品定价策略分析报告

## 1. 市场概况 📊
- **产品总数**: {len(df):,} 个
- **平均折扣率**: {df['real_discount'].mean():.1f}%
- **平均评分**: {df['rating'].mean():.2f} ⭐
- **价格弹性系数**: {elasticity:.2f}

## 2. 情感分析 💭
- **正面评价占比**: {positive_ratio:.1f}%
- **中性评价占比**: {neutral_ratio:.1f}%
- **负面评价占比**: {negative_ratio:.1f}%

## 3. 定价模型表现 🎯
- **模型准确率**: {model_score:.2%}

## 4. 价格优化建议 💡
- **建议提价产品数**: {len(recommendations[recommendations['expected_change'] > 0]):,} 个
- **建议降价产品数**: {len(recommendations[recommendations['expected_change'] < 0]):,} 个
- **预期平均利润提升**: {recommendations['expected_change'].mean():.2f}%

## 5. 重点关注产品 TOP5 ⭐
| 产品ID | 当前价格 (₹) | 建议价格 (₹) | 预期变化 (%) |
|--------|-------------|--------------|--------------|
{table_content}

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