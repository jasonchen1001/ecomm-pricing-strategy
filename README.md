# Cross-border E-commerce Pricing Strategy Optimization / 跨境电商产品定价策略优化

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

[English](#english) | [中文](#chinese)

## English

### Overview
A data-driven pricing strategy optimization system for Amazon India products, focusing on sentiment analysis and dynamic pricing recommendations. The system analyzes customer reviews using BERT model and provides price adjustment suggestions based on sentiment scores and market performance.

### Project Structure
```
amazon_pricing/
├── data/                # Data files
│   ├── amazon.csv      # Raw data
│   └── processed_amazon.csv  # Processed data with sentiment scores
├── src/                 # Source code
│   ├── data_preprocessing.py # Data cleaning and feature extraction
│   ├── sentiment_analysis.py # BERT-based sentiment analysis
│   ├── pricing_model.py      # Random Forest based pricing model
│   └── main.py              # Report generation script
├── outputs/             # Analysis results
│   └── report/         # Generated reports
│       └── pricing_strategy_report.md  # Bilingual analysis report
└── README.md
```

### Key Features
- 📊 Sentiment Analysis
  - BERT-based review sentiment analysis
  - Positive/Negative classification
  - Sentiment score calculation (0-1)
  - Review sentiment distribution analysis

- 💰 Price Optimization
  - Random Forest based pricing model
  - Price adjustments (±5% range)
  - Revenue impact prediction
  - Confidence score for recommendations

- 📝 Strategy Report
  - Bilingual report (English/Chinese)
  - Market overview
  - Sentiment analysis results
  - Top products for price adjustments
  - Strategic recommendations

### Analysis Results
- **Sentiment Distribution**
  - Positive Reviews: 58.3%
  - Negative Reviews: 41.7%
  - Average Sentiment Score: 0.94

- **Price Adjustments**
  - Recommended Increases: 65.3%
  - Recommended Decreases: 34.7%
  - Average Adjustment: 0.9%
  - Expected Revenue Growth: 3.5%

### Dataset
- Amazon India product data (including prices/reviews/ratings)
- **Fields**:
  ```python
  product_id          # Unique product identifier
  product_name        # Product name
  discounted_price    # Current price (₹)
  actual_price        # Original price (₹)
  discount_percentage # Discount rate
  rating             # Average rating (1-5)
  rating_count       # Number of ratings
  review_content     # User review text
  sentiment          # POSITIVE/NEGATIVE
  sentiment_score    # Sentiment score (0-1)
  ```

### Requirements
- Python 3.8+
- Required packages:
  ```
  pandas>=1.3.0
  numpy>=1.19.0
  transformers>=4.5.0
  torch>=1.8.0
  pytz>=2021.1
  scikit-learn>=0.24.0
  ```

### Quick Start
1. Clone the repository
```bash
git clone https://github.com/yourusername/amazon_pricing.git
cd amazon_pricing
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the analysis
```bash
python src/main.py
```

The script will:
1. Load the processed data with sentiment analysis results
2. Generate price adjustment recommendations
3. Create a bilingual analysis report in `outputs/report/`

### Model Details
- **Sentiment Analysis**: DistilBERT model fine-tuned on Amazon reviews
- **Pricing Model**: Random Forest with features:
  - Review sentiment score
  - Rating and rating count
  - Current discount rate
  - Category average price ratio

### License
MIT License

### Changelog

#### [1.0.0] - 2024-01-10

##### Added
- Complete data analysis pipeline
- Interactive data dashboard
- Price elasticity model

##### Optimized
- Improved model accuracy
- Enhanced UI experience

##### Fixed
- Fixed outlier handling in data preprocessing
- Improved sentiment analysis accuracy

### Feedback & Support

#### Contact
- Submit Issue: [GitHub Issues](https://github.com/jasonchen1001/ecomm-pricing-strategy/issues)
- Email: yizhouchen68@gmail.com

#### Get Full Solution
**Optimize pricing strategy, lead the market competition**
[Contact for details](mailto:yizhouchen68@gmail.com)

### View Full Report
For detailed analysis and recommendations, please check the [full report](amazon_pricing/outputs/report/pricing_strategy_report.md).

---

## Chinese

### 概述
基于数据驱动的亚马逊印度产品定价策略优化系统，使用BERT模型进行情感分析，并基于情感得分和市场表现提供价格调整建议。

### 项目结构
```
amazon_pricing/
├── data/                # 数据文件
│   ├── amazon.csv      # 原始数据
│   └── processed_amazon.csv  # 带情感得分的处理后数据
├── src/                 # 源代码
│   ├── data_preprocessing.py # 数据清洗和特征提取
│   ├── sentiment_analysis.py # 基于BERT的情感分析
│   ├── pricing_model.py      # 基于随机森林的定价模型
│   └── main.py              # 报告生成脚本
├── outputs/             # 分析结果
│   └── report/         # 生成的报告
│       └── pricing_strategy_report.md  # 中英双语分析报告
└── README.md
```

### 核心功能
- 📊 情感分析
  - 基于BERT的评论情感分析
  - 正面/负面评论分类
  - 情感得分计算（0-1）
  - 评论情感分布分析

- 💰 价格优化
  - 基于随机森林的定价模型
  - 价格调整建议（±5%范围）
  - 收入影响预测
  - 建议置信度评分

- 📝 策略报告
  - 中英双语报告
  - 市场概况
  - 情感分析结果
  - 重点调价产品
  - 策略建议

### 分析结果
- **情感分布**
  - 正面评价：58.3%
  - 负面评价：41.7%
  - 平均情感得分：0.94

- **价格调整**
  - 建议提价：65.3%
  - 建议降价：34.7%
  - 平均调整：0.9%
  - 预期收入增长：3.5%

### 数据集
- 印度亚马逊产品数据（含价格/评论/评分）
- **字段说明**：
  ```python
  product_id          # 产品唯一标识
  product_name        # 产品名称
  discounted_price    # 当前价格（₹）
  actual_price        # 原价（₹）
  discount_percentage # 折扣率
  rating             # 平均评分（1-5）
  rating_count       # 评分数量
  review_content     # 用户评论文本
  sentiment          # POSITIVE/NEGATIVE
  sentiment_score    # 情感得分（0-1）
  ```

### 环境要求
- Python 3.8+
- 依赖包：
  ```
  pandas>=1.3.0
  numpy>=1.19.0
  transformers>=4.5.0
  torch>=1.8.0
  pytz>=2021.1
  scikit-learn>=0.24.0
  ```

### 快速开始
1. 克隆仓库
```bash
git clone https://github.com/yourusername/amazon_pricing.git
cd amazon_pricing
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行分析
```bash
python src/main.py
```

脚本将：
1. 加载带有情感分析结果的处理后数据
2. 生成价格调整建议
3. 在 `outputs/report/` 创建中英双语分析报告

### 模型详情
- **情感分析**：在亚马逊评论上微调的DistilBERT模型
- **定价模型**：使用以下特征的随机森林：
  - 评论情感得分
  - 评分和评论数量
  - 当前折扣率
  - 类别平均价格比

### 许可证
MIT License

### 更新日志

#### [1.0.0] - 2024-01-10

##### 新增
- 完整的数据分析流程
- 交互式数据看板
- 价格弹性模型

##### 优化
- 提升模型准确率
- 优化UI交互体验

##### 修复
- 修复数据预处理中的异常值处理
- 修复情感分析准确性问题

### 问题反馈

#### 联系方式
- 提交 Issue: [GitHub Issues](https://github.com/jasonchen1001/ecomm-pricing-strategy/issues)
- 邮件: yizhouchen68@gmail.com

#### 获取完整方案
**优化定价策略，领跑市场竞逐**
[联系获取详情](mailto:yizhouchen68@gmail.com)

### 查看完整报告
详细的分析结果和建议请查看[完整报告](amazon_pricing/outputs/report/pricing_strategy_report.md)。