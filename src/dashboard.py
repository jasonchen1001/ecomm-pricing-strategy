import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, extract_features
from price_elasticity import PriceElasticityAnalyzer
from sentiment_analysis import SentimentAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from wordcloud import WordCloud
from nltk.corpus import opinion_lexicon
import nltk

# 必须是第一个 Streamlit 命令
st.set_page_config(page_title="Amazon Product Analysis", layout="wide")

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stPlotly {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 确保下载必要的词典
try:
    nltk.data.find('corpora/opinion_lexicon')
except LookupError:
    nltk.download('opinion_lexicon')

def normalize_sizes(sizes, min_size=8, max_size=40):
    """将词云字体大小归一化到合理范围"""
    if len(sizes) == 0:
        return []
    min_val = min(sizes)
    max_val = max(sizes)
    if max_val == min_val:
        return [max_size] * len(sizes)
    return [min_size + (max_size - min_size) * (s - min_val) / (max_val - min_val) for s in sizes]

def get_color_gradient(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    """为词云生成渐变色"""
    # 为积极评论使用绿色渐变
    positive_colors = ['#90EE90', '#32CD32', '#228B22', '#006400']  # 浅绿到深绿
    # 为消极评论使用红色渐变
    negative_colors = ['#FFB6C1', '#FF6B6B', '#DC143C', '#8B0000']  # 浅红到深红
    
    # 根据字体大小选择颜色
    colors = positive_colors if random_state.randint(2) == 0 else negative_colors
    color_idx = int(font_size * (len(colors) - 1) / 100)
    return colors[min(color_idx, len(colors) - 1)]

def create_wordcloud(text, title, sentiment_type='positive'):
    """创建词云图"""
    # 获取NLTK的情感词典
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    
    # 产品评论特定的积极词
    product_specific_positive = {
        'affordable', 'worth', 'sturdy', 'durable', 'reliable', 
        'fast', 'quick', 'solid', 'perfect', 'excellent',
        'strong', 'stable', 'premium', 'professional', 'recommended',
        'satisfied', 'quality', 'great', 'nice', 'good',
        'convenient', 'efficient', 'effective', 'impressive'
    }
    
    # 产品评论特定的消极词
    product_specific_negative = {
        'defective', 'faulty', 'broken', 'useless',
        'disappointing', 'terrible', 'horrible', 'awful',
        'worthless', 'poor-quality', 'unreliable', 'unstable',
        'overpriced', 'ineffective', 'malfunctioning',
        'negative', 'strain'
    }
    
    # 需要移除的歧义词
    ambiguous_words = {
        'quality',    # 可能表示好(积极)或差(消极)
        'cheap',      # 可能表示便宜(积极)或劣质(消极)
        'hard',       # 可能表示坚硬(积极)或困难(消极)
        'emergency',  # 描述情况而非产品质量
        'blame',      # 描述行为而非产品
        'basic',      # 可能是中性描述
        'simple',     # 可能是积极或消极
        'just',       # 通常是中性词
        'want',       # 意愿描述
        'need',       # 需求描述
        'try',        # 行为描述
        'return',     # 行为描述
        'cost',       # 价格描述
        'price',      # 价格描述
        'charge'      # 可能是充电或收费
    }
    
    # 停用词
    stop_words = {
        'issues', 'issue', 'problem', 'problems',  # 中性词
        'like', 'well', 'better', 'works', 'work',  # 中性/描述性词
        'cable', 'charger', 'wire', 'cord', 'adapter',  # 产品相关词
        'time', 'month', 'day', 'year', 'week',  # 时间相关词
        'amazon', 'product', 'purchase', 'bought', 'order',  # 购买相关词
        'use', 'using', 'used', 'usage',  # 使用相关词
        'one', 'two', 'three', 'first', 'second',  # 数字相关词
        'the', 'and', 'for', 'that', 'this', 'with',  # 常见停用词
        'was', 'is', 'are', 'were', 'been', 'be', 'have'
    }
    
    # 更新情感词典
    positive_words.update(product_specific_positive)
    negative_words.update(product_specific_negative)
    
    # 移除歧义词和停用词
    positive_words = positive_words - ambiguous_words - stop_words
    negative_words = negative_words - ambiguous_words - stop_words
    
    # 分词并过滤
    words = text.lower().split()
    if sentiment_type == 'positive':
        filtered_words = [word for word in words 
                         if word in positive_words]
        colormap = 'YlGn'
    else:
        filtered_words = [word for word in words 
                         if word in negative_words]
        colormap = 'RdPu'
    
    # 如果没有找到情感词，返回空图
    if not filtered_words:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No significant sentiment words found',
                ha='center', va='center')
        ax.axis('off')
        return fig
    
    # 生成词云
    text = ' '.join(filtered_words)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=30,
        min_font_size=12,
        max_font_size=160,
        prefer_horizontal=0.7
    ).generate(text)
    
    # 创建图形但不添加标题
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_top_sentiment_words(text, sentiment_type='positive', n=10):
    """获取前N个最常见的情感词"""
    # 获取情感词典
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    
    # 产品评论特定的积极词
    product_specific_positive = {
        'affordable', 'worth', 'sturdy', 'durable', 'reliable', 
        'fast', 'quick', 'solid', 'perfect', 'excellent',
        'strong', 'stable', 'premium', 'professional', 'recommended',
        'satisfied', 'quality', 'great', 'nice', 'good',
        'convenient', 'efficient', 'effective', 'impressive'
    }
    
    # 产品评论特定的消极词（只保留明确的负面词）
    product_specific_negative = {
        'defective', 'faulty', 'broken', 'useless',
        'disappointing', 'terrible', 'horrible', 'awful',
        'worthless', 'poor-quality', 'unreliable', 'unstable',
        'overpriced', 'ineffective', 'malfunctioning'
    }
    
    # 需要从情感词典中移除的歧义词
    ambiguous_words = {
        'quality',    # 可能表示便宜(积极)或劣质(消极)
        'cheap',      # 可能表示便宜(积极)或劣质(消极)
        'hard',       # 可能表示坚硬(积极)或困难(消极)
        'emergency',  # 描述情况而非产品质量
        'blame',      # 描述行为而非产品
        'basic',      # 可能是中性描述
        'simple',     # 可能是积极或消极
        'just',       # 通常是中性词
        'want',       # 意愿描述
        'need',       # 需求描述
        'try',        # 行为描述
        'return',     # 行为描述
        'cost',       # 价格描述
        'price',      # 价格描述
        'charge'      # 可能是充电或收费
    }
    
    # 额外的停用词
    extra_stop_words = {
        'issues', 'issue', 'problem', 'problems',  # 中性词
        'like', 'well', 'better', 'works', 'work',  # 中性/描述性词
        'cable', 'charger', 'wire', 'cord', 'adapter',  # 产品相关词
        'time', 'month', 'day', 'year', 'week',  # 时间相关词
        'amazon', 'product', 'purchase', 'bought', 'order',  # 购买相关词
        'use', 'using', 'used', 'usage',  # 使用相关词
        'one', 'two', 'three', 'first', 'second',  # 数字相关词
    }
    
    # 更新情感词典
    positive_words.update(product_specific_positive)
    negative_words.update(product_specific_negative)
    
    # 移除所有歧义词和停用词
    positive_words = positive_words - ambiguous_words - extra_stop_words
    negative_words = negative_words - ambiguous_words - extra_stop_words
    
    # 分词并过滤
    words = text.lower().split()
    word_freq = {}
    target_words = positive_words if sentiment_type == 'positive' else negative_words
    
    for word in words:
        if word in target_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 获取前N个最常见的词
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_words

def get_display_text(en_text, zh_text, lang='en'):
    """根据选择的语言返回显示文本"""
    return en_text if lang == 'en' else zh_text

def main():
    # 添加语言选择器到侧边栏
    lang = st.sidebar.selectbox(
        "Language / 语言",
        options=['en', 'zh'],
        format_func=lambda x: "English" if x == 'en' else "中文"
    )
    
    try:
        df = load_data('amazon.csv')
    except Exception as e:
        st.error(get_display_text(
            'Error loading data: Please check if amazon.csv exists in the correct location.',
            '加载数据错误：请检查 amazon.csv 文件是否存在于正确位置。',
            lang
        ))
        st.exception(e)
        return
    
    st.title(get_display_text(
        '🚗 Amazon Product Analysis Dashboard',
        '🚗 亚马逊产品分析仪表板',
        lang
    ))
    
    # Market Overview 部分移到这里
    st.header(get_display_text('📊 Market Overview', '📊 市场概览', lang))
    
    # 显示关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            get_display_text("Total Products", "产品总数", lang),
            f"{len(df):,}"
        )
    
    with col2:
        avg_rating = df['rating'].mean()
        st.metric(
            get_display_text("Average Rating", "平均评分", lang),
            f"{avg_rating:.2f} ⭐"
        )
    
    with col3:
        avg_price = df['discounted_price'].mean()
        st.metric(
            get_display_text("Average Price", "平均价格", lang),
            f"₹{avg_price:.2f}"
        )
    
    with col4:
        avg_discount = df['real_discount'].mean()
        st.metric(
            get_display_text("Average Discount", "平均折扣", lang),
            f"{avg_discount:.1f}%"
        )
    
    # 创建标签页 - 调整顺序
    tab1, tab2, tab3 = st.tabs([
        get_display_text('Price Analysis', '价格分析', lang),
        get_display_text('Review Analysis', '评论分析', lang),
        get_display_text('Product Rankings', '产品排名', lang)
    ])
    
    # 加载数据时显示进度条
    with st.spinner('Loading data...'):
        df, features, analyzer = load_cached_data()
    
    # 创建价格弹性分析器实例
    elasticity_analyzer = PriceElasticityAnalyzer()
    
    # 侧边栏优化
    st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/4/4a/Amazon_icon.svg', width=100)
    st.sidebar.title(get_display_text('Analysis Controls', '分析控制', lang))
    
    # 简化侧边栏设置
    st.sidebar.markdown("---")
    st.sidebar.subheader(get_display_text('Price Elasticity Settings', '价格弹性设置', lang))
    
    # 只保留弹性系数计算方法选择
    elasticity_method = st.sidebar.selectbox(
        get_display_text('Calculation Method', '计算方法', lang),
        ['Log-Log', 'Point', 'Arc'],
        help=get_display_text(
            "Method to calculate price elasticity",
            "计算价格弹性的方法",
            lang
        )
    )
    
    # 添加更多筛选器
    price_range = st.sidebar.slider(
        get_display_text('Price Range (₹)', '价格范围 (₹)', lang),
        float(df['discounted_price'].min()),
        float(df['discounted_price'].max()),
        (float(df['discounted_price'].min()), float(df['discounted_price'].max()))
    )
    
    rating_filter = st.sidebar.slider(
        get_display_text('Minimum Rating', '最低评分', lang),
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5
    )
    
    # 数据筛选
    mask = (
        (df['discounted_price'] >= price_range[0]) & 
        (df['discounted_price'] <= price_range[1]) &
        (df['rating'] >= rating_filter)
    )
    filtered_df = df[mask]
    
    # 添加刷新按钮
    if st.sidebar.button(get_display_text('Refresh Analysis', '刷新分析', lang)):
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except AttributeError:
                st.error(get_display_text(
                    "Refresh functionality not available in this Streamlit version",
                    "此版本的Streamlit不支持刷新功能",
                    lang
                ))
    
    # 使用tabs组织内容
    with tab1:
        # Price Analysis 内容
        #st.header(get_display_text('💰 Price Analysis', '💰 价格分析', lang))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(get_display_text('💰 Price Distribution', '💰 价格分布', lang))
            fig = px.histogram(
                filtered_df,
                x='discounted_price',
                nbins=50,
                title=get_display_text(' Product Price Distribution', ' 产品价格分布', lang)
            )
            
            # 更新图表布局
            fig.update_layout(
                bargap=0.2,  # 添加柱子之间的间隔
                plot_bgcolor='white',  # 设置白色背景
                showlegend=False,
                xaxis=dict(
                    title='Price (₹)',
                    gridcolor='lightgrey',
                    showgrid=True,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickformat='₹%d'
                ),
                yaxis=dict(
                    title='Number of Products',
                    gridcolor='lightgrey',
                    showgrid=True,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                margin=dict(l=40, r=40, t=40, b=40)  # 调整边距
            )
            
            # 更新柱子颜色和边框
            fig.update_traces(
                marker_color='rgb(30, 144, 255)',  # 设置柱子颜色为深蓝色
                marker_line_color='rgb(8, 48, 107)',  # 设置边框颜色
                marker_line_width=1  # 设置边框宽度
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 添加相关性分析
            st.markdown("---")
            st.subheader(get_display_text('📊 Correlation Analysis', '📊 相关性分析', lang))
            correlation_matrix = filtered_df[
                ['discounted_price', 'rating', 'rating_count', 'real_discount']
            ].corr()
            
            # 双语标签
            labels = {
                'discounted_price': get_display_text('Price', '价格', lang),
                'rating': get_display_text('Rating', '评分', lang),
                'rating_count': get_display_text('Reviews', '评论数', lang),
                'real_discount': get_display_text('Discount', '折扣', lang)
            }
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=list(labels.values()),
                y=list(labels.values()),
                text=correlation_matrix.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate=get_display_text(
                    '%{x} vs %{y}<br>Correlation: %{z:.2f}',
                    '%{x} vs %{y}<br>相关性: %{z:.2f}',
                    lang
                ) + '<extra></extra>',
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title=get_display_text('Correlation Matrix', '相关性矩阵', lang),
                height=400,
                hoverlabel=dict(bgcolor="white"),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(get_display_text('📈 Price Elasticity Analysis', '📈 价格弹性分析', lang))
            
            # 计算价格弹性系数
            elasticity = elasticity_analyzer.calculate_elasticity(
                filtered_df['discounted_price'].values,
                filtered_df['rating_count'].values,
                method=elasticity_method
            )
            
            # 显示弹性系数及其含义
            st.metric(get_display_text('Price Elasticity', '价格弹性系数', lang), f"{elasticity:.2f}")
            
            if elasticity < 0.5:
                st.success(get_display_text(
                    '''
                    **Low Price Elasticity** (< 0.5):
                    - Consumers are less sensitive to price changes
                    - Consider appropriate price increases
                    - Focus on product quality and brand building
                    ''',
                    '''
                    **低价格弹性** (< 0.5):
                    - 消费者对价格变化不敏感
                    - 可以考虑适当提高价格
                    - 重点关注产品质量和品牌建设
                    ''',
                    lang
                ))
            else:
                st.warning(get_display_text(
                    '''
                    **High Price Elasticity** (≥ 0.5):
                    - Consumers are sensitive to price changes
                    - Need careful price adjustments
                    - Pay attention to competitor pricing
                    ''',
                    '''
                    **高价格弹性** (≥ 0.5):
                    - 消费者对价格变化敏感
                    - 需要谨慎调整价格
                    - 关注竞品定价策略
                    ''',
                    lang
                ))
            
            # 根据不同的计算方法显示不同的价格-需求关系图
            if elasticity_method == 'Log-Log':
                # 对数转换后的散点图
                fig = px.scatter(
                    filtered_df,
                    x='discounted_price',
                    y='rating_count',
                    title=get_display_text(
                        'Price-Demand Relationship',
                        '价格-需求关系图',
                        lang
                    ),
                    labels={
                        'discounted_price': get_display_text('Price (₹)', '价格 (₹)', lang),
                        'rating_count': get_display_text('Demand (log)', '需求量 (对数)', lang)
                    }
                )
                
                # 添加趋势线
                fig.add_traces(go.Scatter(
                    x=filtered_df['discounted_price'],
                    y=filtered_df['rating_count'].mean() * np.ones(len(filtered_df)),
                    mode='lines',
                    name=get_display_text('Trend Line', '趋势线', lang),
                    line=dict(color='red', dash='dash')
                ))
            elif elasticity_method == 'Point':
                # 分段点弹性图
                fig = px.scatter(
                    filtered_df,
                    x='discounted_price',
                    y='rating_count',
                    title='Point Price Elasticity',
                    labels={
                        'discounted_price': 'Price (₹)',
                        'rating_count': 'Demand (Reviews)'
                    }
                )
                # 添加分段点弹性线
                sorted_df = filtered_df.sort_values('discounted_price')
                segments = np.array_split(sorted_df, 5)
                for segment in segments:
                    fig.add_trace(go.Scatter(
                        x=segment['discounted_price'],
                        y=segment['rating_count'],
                        mode='lines',
                        name=f'Segment {len(fig.data)}'
                    ))
            else:  # Arc
                # 弧弹性图
                fig = px.scatter(
                    filtered_df,
                    x='discounted_price',
                    y='rating_count',
                    title='Arc Price Elasticity',
                    labels={
                        'discounted_price': 'Price (₹)',
                        'rating_count': 'Demand (Reviews)'
                    },
                    trendline="lowess"  # 使用局部加权回归
                )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Review Analysis 标签页
        st.subheader(get_display_text('Sentiment Analysis', '情感分析', lang))
        
        # 获取情感分析摘要
        sentiment_summary = analyzer.get_sentiment_summary(filtered_df, lang)
        
        # 显示情感分布
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                get_display_text('Positive Reviews', '正面评论', lang),
                f"{sentiment_summary['positive_pct']:.1f}%"
            )
        with col2:
            st.metric(
                get_display_text('Neutral Reviews', '中性评论', lang),
                f"{sentiment_summary['neutral_pct']:.1f}%"
            )
        with col3:
            st.metric(
                get_display_text('Negative Reviews', '负面评论', lang),
                f"{sentiment_summary['negative_pct']:.1f}%"
            )
        
        st.plotly_chart(sentiment_summary['distribution_plot'], use_container_width=True)
        
        # 显示Top情感词
        col1, col2 = st.columns(2)
        with col1:
            pos_words = analyzer.get_top_sentiment_words(filtered_df, 'positive', 10, lang)
            st.plotly_chart(pos_words['plot'], use_container_width=True)
        
        with col2:
            neg_words = analyzer.get_top_sentiment_words(filtered_df, 'negative', 10, lang)
            st.plotly_chart(neg_words['plot'], use_container_width=True)
    
    with tab3:
        # Product Rankings 内容
        st.header(get_display_text('🏆 Product Rankings', '🏆 产品排名', lang))
        top_products = filtered_df.nlargest(10, 'rating')[
            ['product_name', 'discounted_price', 'rating', 'rating_count']
        ].reset_index(drop=True)
        
        # 使用更好的表格展示
        st.dataframe(
            top_products,
            column_config={
                "product_name": "Product Name",
                "discounted_price": st.column_config.NumberColumn(
                    get_display_text("Price (₹)", "价格 (₹)", lang),
                    format="₹%.2f"
                ),
                "rating": st.column_config.NumberColumn(
                    get_display_text("Rating", "评分", lang),
                    format="%.1f ⭐"
                ),
                "rating_count": st.column_config.NumberColumn(
                    get_display_text("Reviews", "评论", lang),
                    format="%d 📝"
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    # 移除页面底部的相关性矩阵
    # 只保留页脚
    st.markdown("""---""")
    st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ by Yanzhen Chen | Data last updated: 2025</p>
        </div>
    """, unsafe_allow_html=True)

# 加载数据
@st.cache_data
def load_cached_data():
    df = load_data('amazon.csv')
    features = extract_features(df)
    
    # 情感分析
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_reviews(df)
    
    return df, features, analyzer

if __name__ == '__main__':
    main() 