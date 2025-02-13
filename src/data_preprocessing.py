import pandas as pd

def load_data(file_path):
    """加载数据并进行基础清洗"""
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
    
    # 只保留线缆类产品
    cables_df = df[df['category'].str.contains('Cables', na=False)]
    
    return cables_df

def extract_features(df):
    """特征工程"""
    features = pd.DataFrame()
    
    # 基础特征
    features['price'] = df['discounted_price']
    features['rating'] = df['rating']
    features['rating_count'] = df['rating_count']
    features['discount'] = df['real_discount']
    
    # 价格区间
    features['price_segment'] = pd.qcut(features['price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # 销量代理指标(评论数)
    features['popularity'] = pd.qcut(features['rating_count'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    return features 

def analyze_price_sensitivity(df):
    """分析价格敏感性"""
    # 计算价格弹性系数
    df['price_elasticity'] = df['price'].diff() / df['price']
    
    return df