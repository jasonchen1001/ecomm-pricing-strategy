import pandas as pd
import re
import numpy as np

def clean_text(text):
    """清理文本数据"""
    if not isinstance(text, str):
        return ''
    
    # 转换为小写
    text = text.lower()
    
    # 移除 URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 替换逗号为空格
    text = text.replace(',', ' ')
    
    # 移除特殊字符和标点符号，但保留基本标点
    text = re.sub(r'[^\w\s.!?]', ' ', text)
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除数字
    text = re.sub(r'\d+', '', text)
    
    # 移除 HTML 标签
    text = re.sub(r'<.*?>', '', text)  

    # 移除非字母和空格的字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    
    return text.strip()

def load_data(file_path):
    """加载数据并进行基础清洗"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 处理价格列
    df['discounted_price'] = df['discounted_price'].str.replace('₹','').str.replace(',','').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹','').str.replace(',','').astype(float)
    
    # 处理评论数列
    df['rating_count'] = df['rating_count'].str.replace(',','').astype(float)
    
    # 处理评分列
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # 计算真实折扣率
    df['real_discount'] = df['discount_percentage'].str.rstrip('%').astype(float)
    
    # 添加产品类别分类
    df['main_category'] = df['category'].str.split('|').str[0]
    
    # 清理评论文本
    df['review_title'] = df['review_title'].fillna('')
    df['review_content'] = df['review_content'].fillna('')
    df['cleaned_review'] = (df['review_title'] + ' ' + df['review_content']).apply(clean_text)
    
    # 清理产品描述
    df['about_product'] = df['about_product'].fillna('')
    df['cleaned_about'] = df['about_product'].apply(clean_text)
    
    # 处理数值列的缺失值和异常值
    # 使用中位数填充数值型特征的缺失值
    numeric_cols = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'real_discount']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 移除异常值（价格和评论数为0或极端值的记录）
    df = df[
        (df['discounted_price'] > 0) & 
        (df['actual_price'] > 0) & 
        (df['rating_count'] > 0) &
        (df['rating'].between(1, 5))  # 评分应该在1-5之间
    ]
    
    # 移除清理后为空的记录
    df = df[
        (df['cleaned_review'].str.len() > 0) & 
        (df['cleaned_about'].str.len() > 0)
    ]
    
    return df

def extract_features(df):
    """特征工程"""
    features = pd.DataFrame()
    
    # 提取基础数值特征
    features['price'] = df['discounted_price']
    features['rating'] = df['rating']
    features['rating_count'] = df['rating_count']
    features['discount'] = df['real_discount']
    features['main_category'] = df['main_category']
    
    # 创建分箱函数：将数据分成5个等级
    def safe_qcut(x):
        try:
            # 尝试创建等频分箱（每个区间数据量相等）
            return pd.qcut(x, q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        except ValueError:
            # 处理特殊情况
            if len(x.unique()) == 1:  # 如果所有值都相同
                return pd.Series(['medium'] * len(x), index=x.index)
            # 如果不同值的数量少于5个
            n_unique = min(len(x.unique()), 5)
            try:
                return pd.qcut(x, q=n_unique, labels=['very_low', 'low', 'medium', 'high', 'very_high'][:n_unique])
            except ValueError:
                # 如果等频分箱失败，使用等宽分箱
                return pd.cut(x, bins=n_unique, labels=['very_low', 'low', 'medium', 'high', 'very_high'][:n_unique])
    
    # 创建价格区间特征（按类别分组后计算相对价格水平）
    df['price_segment'] = df.groupby('main_category')['discounted_price'].transform(safe_qcut)
    features['price_segment'] = df['price_segment']
    
    # 创建受欢迎程度特征（按类别分组后计算相对评论数量水平）
    df['popularity'] = df.groupby('main_category')['rating_count'].transform(safe_qcut)
    features['popularity'] = df['popularity']
    
    return features

def get_category_stats(df):
    """获取各类别的统计信息"""
    stats = df.groupby('main_category').agg({
        'discounted_price': ['count', 'mean', 'std', 'min', 'max'],
        'rating': ['mean', 'std'],
        'rating_count': ['sum', 'mean'],
        'real_discount': ['mean', 'std']
    }).round(2)
    
    # 添加产品数量占比
    total_products = df['main_category'].count()
    stats['product_percentage'] = (stats[('discounted_price', 'count')] / total_products * 100).round(2)
    
    return stats

def main():
    """测试数据处理功能"""
    # 测试数据加载和清理
    print("=== 测试数据加载和清理 ===")
    try:
        df = load_data('data/amazon.csv')
        print("\n数据样例:")
        print(df[['product_name', 'main_category', 'discounted_price', 'cleaned_review']].head())
        
        # 测试特征提取
        print("\n=== 测试特征提取 ===")
        features = extract_features(df)
        print("\n特征样例:")
        print(features.head())
        
        # 测试类别统计
        print("\n=== 测试类别统计 ===")
        stats = get_category_stats(df)
        print("\n类别统计:")
        print(stats.head())
        
        # 保存处理后的数据
        print("\n=== 保存处理后的数据 ===")
        df.to_csv('data/processed_amazon.csv', index=False)
        print("数据已保存到 data/processed_amazon.csv")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请确保数据文件位于 data/amazon.csv")

if __name__ == "__main__":
    main()
