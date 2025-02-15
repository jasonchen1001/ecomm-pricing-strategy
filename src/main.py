import pandas as pd
import os
from datetime import datetime
import pytz

# 项目路径配置
REPORT_DIR = '../outputs'
# 创建必要的目录
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_report():
    """生成中英文分析报告"""
    try:
        # 加载数据
        df = pd.read_csv('data/processed_amazon.csv')
        recommendations = pd.read_csv('data/price_recommendations.csv')
        
        # 计算基本统计信息
        total_products = len(recommendations)
        
        # 价格调整统计
        price_changes = recommendations['adjusted_change']
        increase_mask = price_changes > 0
        decrease_mask = price_changes < 0
        no_change_mask = abs(price_changes) < 3
        
        increases = sum(increase_mask)
        decreases = sum(decrease_mask)
        no_changes = sum(no_change_mask)
        
        # 收入影响
        current_revenue = recommendations['current_revenue'].sum()
        expected_revenue = recommendations['expected_revenue'].sum()
        revenue_change = ((expected_revenue - current_revenue) / current_revenue * 100)
        
        # 情感分析统计
        total_reviews = len(df)
        positive_reviews = sum(df['sentiment'] == 'POSITIVE')
        negative_reviews = sum(df['sentiment'] == 'NEGATIVE')
        avg_sentiment = df['sentiment_score'].mean()
        
        # 获取北京时间
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = datetime.now(beijing_tz)
        
        # 生成报告
        report_path = os.path.join(REPORT_DIR, 'pricing_strategy_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            # 英文部分
            f.write(f"""# Cross-border E-commerce Pricing Strategy Optimization / 跨境电商产品定价策略优化

[English](#english) | [中文](#chinese)

## English

### Amazon Product Pricing Strategy Analysis Report

#### 1. Market Overview 📊
- **Total Products Analyzed**: {total_products:,}
- **Average Rating**: {df['rating'].mean():.2f} ⭐
- **Average Discount**: {df['discount_percentage'].str.rstrip('%').astype(float).mean():.1f}%

#### 2. Sentiment Analysis 💭
##### Overall Sentiment Distribution
- **Total Reviews**: {total_reviews:,}
- **Positive Reviews**: {positive_reviews:,} ({positive_reviews/total_reviews*100:.1f}%)
- **Negative Reviews**: {negative_reviews:,} ({negative_reviews/total_reviews*100:.1f}%)
- **Positive to Negative Ratio**: {positive_reviews}:{negative_reviews} ({positive_reviews/negative_reviews:.2f}:1)
- **Average Sentiment Score**: {avg_sentiment:.2f}

##### Sentiment Distribution Characteristics
- Overall positive sentiment, with more than half of the reviews being positive
- High sentiment score indicates good user satisfaction
- Need to monitor negative reviews for timely improvements

#### 3. Price Adjustment Suggestions 💰
##### Price Adjustment Distribution
- **Recommended Increases**: {increases:,} products ({increases/total_products*100:.1f}%)
- **Recommended Decreases**: {decreases:,} products ({decreases/total_products*100:.1f}%)
- **Maintain Current Price**: {no_changes:,} products ({no_changes/total_products*100:.1f}%)

##### Adjustment Range
- **Maximum Increase**: {price_changes.max():.1f}%
- **Maximum Decrease**: {price_changes.min():.1f}%
- **Average Adjustment**: {price_changes.mean():.1f}%

##### Revenue Impact
- **Current Total Revenue**: ₹{current_revenue:,.2f}
- **Expected Total Revenue**: ₹{expected_revenue:,.2f}
- **Expected Growth**: {revenue_change:.1f}%

#### 4. Key Products to Watch ⭐

##### Top Price Increases (Top 5)""")

            # 添加英文版最大提价产品
            top_increases = recommendations.nlargest(5, 'adjusted_change')
            for _, row in top_increases.iterrows():
                product = df[df['product_id'] == row['product_id']].iloc[0]
                f.write(f"""
- **{product['product_name'][:50]}...**
  - Current Price: ₹{row['current_price']:.2f}
  - Recommended Price: ₹{row['recommended_price']:.2f} (+{row['adjusted_change']:.1f}%)
  - Rating: {product['rating']}⭐ ({product['rating_count']} Reviews)
  - Sentiment Score: {product['sentiment_score']:.2f}""")

            f.write("\n\n##### Top Price Decreases (Top 5)")
            # 添加英文版最大降价产品
            top_decreases = recommendations.nsmallest(5, 'adjusted_change')
            for _, row in top_decreases.iterrows():
                product = df[df['product_id'] == row['product_id']].iloc[0]
                f.write(f"""
- **{product['product_name'][:50]}...**
  - Current Price: ₹{row['current_price']:.2f}
  - Recommended Price: ₹{row['recommended_price']:.2f} ({row['adjusted_change']:.1f}%)
  - Rating: {product['rating']}⭐ ({product['rating_count']} Reviews)
  - Sentiment Score: {product['sentiment_score']:.2f}""")

            f.write(f"""

#### 5. Strategic Recommendations 📈

##### Pricing Strategy
1. **Differential Pricing**
   - Adjust prices based on ratings and sentiment
   - Moderate price increases for high-rated products
   - Consider promotions for low-rated products

2. **Cautious Adjustment**
   - Recommend small adjustments for most products
   - Monitor user feedback after price changes
   - Regular evaluation of pricing strategy

3. **Key Focus Areas**
   - Monitor products with negative reviews
   - Optimize service for products with price increases
   - Track competitor pricing changes

##### Improvement Suggestions
1. **Product Quality**
   - Focus on improving products with negative reviews
   - Maintain advantages of high-rated products
   - Continuous improvement of user experience

2. **Service Optimization**
   - Strengthen after-sales support
   - Improve logistics efficiency
   - Enhance user feedback handling

3. **Marketing Strategy**
   - Highlight advantages of high-rated products
   - Targeted promotions for low-rated products
   - Strengthen brand image

## Chinese

### 亚马逊产品定价策略分析报告

#### 1. 市场概况 📊
- **分析产品总数**: {total_products:,}
- **平均评分**: {df['rating'].mean():.2f} ⭐
- **平均折扣率**: {df['discount_percentage'].str.rstrip('%').astype(float).mean():.1f}%

#### 2. 情感分析 💭
##### 总体情感分布
- **评论总数**: {total_reviews:,}
- **正面评价**: {positive_reviews:,} ({positive_reviews/total_reviews*100:.1f}%)
- **负面评价**: {negative_reviews:,} ({negative_reviews/total_reviews*100:.1f}%)
- **正负比例**: {positive_reviews}:{negative_reviews} ({positive_reviews/negative_reviews:.2f}:1)
- **平均情感得分**: {avg_sentiment:.2f}

##### 情感分布特点
- 总体评价偏正面，正面评价占比超过半数
- 情感得分较高，表明用户满意度良好
- 需关注负面评价，及时改进产品和服务

#### 3. 价格调整建议 💰
##### 调价分布
- **建议提价**: {increases:,} 个产品 ({increases/total_products*100:.1f}%)
- **建议降价**: {decreases:,} 个产品 ({decreases/total_products*100:.1f}%)
- **维持现价**: {no_changes:,} 个产品 ({no_changes/total_products*100:.1f}%)

##### 调价幅度
- **最大提价**: {price_changes.max():.1f}%
- **最大降价**: {price_changes.min():.1f}%
- **平均调整**: {price_changes.mean():.1f}%

##### 收入影响
- **当前总收入**: ₹{current_revenue:,.2f}
- **预期总收入**: ₹{expected_revenue:,.2f}
- **预期增长**: {revenue_change:.1f}%

#### 4. 重点关注产品 ⭐

##### 最大提价产品 (Top 5)""")

            # 添加中文版最大提价产品
            for _, row in top_increases.iterrows():
                product = df[df['product_id'] == row['product_id']].iloc[0]
                f.write(f"""
- **{product['product_name'][:50]}...**
  - 当前价格: ₹{row['current_price']:.2f}
  - 建议价格: ₹{row['recommended_price']:.2f} (+{row['adjusted_change']:.1f}%)
  - 评分: {product['rating']}⭐ ({product['rating_count']} 评论)
  - 情感得分: {product['sentiment_score']:.2f}""")

            f.write("\n\n##### 最大降价产品 (Top 5)")
            # 添加中文版最大降价产品
            for _, row in top_decreases.iterrows():
                product = df[df['product_id'] == row['product_id']].iloc[0]
                f.write(f"""
- **{product['product_name'][:50]}...**
  - 当前价格: ₹{row['current_price']:.2f}
  - 建议价格: ₹{row['recommended_price']:.2f} ({row['adjusted_change']:.1f}%)
  - 评分: {product['rating']}⭐ ({product['rating_count']} 评论)
  - 情感得分: {product['sentiment_score']:.2f}""")

            f.write(f"""

#### 5. 策略建议 📈

##### 定价策略
1. **差异化定价**
   - 根据产品评分和评论情感调整价格
   - 高评分高情感产品可适度提价
   - 低评分产品考虑降价促销

2. **谨慎调整**
   - 大多数产品建议小幅调整
   - 关注调价后的用户反馈
   - 定期评估价格策略效果

3. **重点关注**
   - 监控负面评价产品
   - 优化高调价产品的服务
   - 跟踪竞品定价变化

##### 改进建议
1. **产品质量**
   - 重点改进负面评价产品
   - 保持高评分产品优势
   - 持续提升用户体验

2. **服务优化**
   - 加强售后支持
   - 提升物流效率
   - 改善用户反馈处理

3. **营销策略**
   - 突出高评分产品优势
   - 针对性促销低评分产品
   - 加强品牌形象建设

---
*Report Generation Time / 报告生成时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')}*
""")
        
        print(f"报告已生成到 {report_path}")
        
    except Exception as e:
        print(f"生成报告时出错: {str(e)}")

if __name__ == "__main__":
    generate_report() 