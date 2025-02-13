import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_data, extract_features
from price_elasticity import PriceElasticityAnalyzer
from sentiment_analysis import SentimentAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
from io import BytesIO
from nltk.corpus import opinion_lexicon
import nltk

# 设置页面配置
st.set_page_config(
    page_title="Amazon Cable Products Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        'cheap',      # 可能表示便宜(积极)或劣质(消极)
        'quality',    # 可能表示便宜(积极)或劣质(消极)
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
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
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

def main():
    # 标题和介绍
    st.title('📊 Amazon Cable Products Pricing Analysis')
    st.markdown("""
    This dashboard provides comprehensive analysis of cable products pricing on Amazon India.
    Use the filters in the sidebar to explore different price ranges and product categories.
    """)
    
    # 加载数据时显示进度条
    with st.spinner('Loading data...'):
        df, features, analyzer = load_cached_data()
    
    # 创建价格弹性分析器实例
    elasticity_analyzer = PriceElasticityAnalyzer()
    
    # 侧边栏优化
    st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/4/4a/Amazon_icon.svg', width=100)
    st.sidebar.title('Analysis Controls')
    
    # 简化侧边栏设置
    st.sidebar.markdown("---")
    st.sidebar.subheader("Price Elasticity Settings")
    
    # 只保留弹性系数计算方法选择
    elasticity_method = st.sidebar.selectbox(
        'Calculation Method',
        ['Log-Log', 'Point', 'Arc'],
        help="Method to calculate price elasticity"
    )
    
    # 添加更多筛选器
    price_range = st.sidebar.slider(
        'Price Range (₹)',
        float(df['discounted_price'].min()),
        float(df['discounted_price'].max()),
        (float(df['discounted_price'].min()), float(df['discounted_price'].max()))
    )
    
    rating_filter = st.sidebar.slider(
        'Minimum Rating',
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
    if st.sidebar.button('Refresh Analysis'):
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()  # 兼容旧版本
            except AttributeError:
                st.error("Refresh functionality not available in this Streamlit version")
    
    # 市场概览使用卡片式设计
    st.header('📈 Market Overview')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            'Total Products',
            len(filtered_df),
            delta=f"{len(filtered_df)-len(df)} from total"
        )
    with col2:
        st.metric(
            'Average Rating',
            f"{filtered_df['rating'].mean():.2f}",
            delta=f"{(filtered_df['rating'].mean() - df['rating'].mean()):.2f}"
        )
    with col3:
        st.metric(
            'Average Discount',
            f"{filtered_df['real_discount'].mean():.1f}%"
        )
    with col4:
        st.metric(
            'Price Range',
            f"₹{filtered_df['discounted_price'].min():.0f} - ₹{filtered_df['discounted_price'].max():.0f}"
        )
    
    # 使用tabs组织内容
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Sentiment Analysis", "Product Rankings"])
    
    with tab1:
        # Price Analysis 标签页
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Price Distribution')
            fig = px.histogram(
                filtered_df,
                x='discounted_price',
                nbins=30,
                title='Price Distribution',
                labels={'discounted_price': 'Price (₹)', 'count': 'Count'},
                hover_data=['discounted_price'],
                opacity=0.7,  # 调整透明度
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
            st.subheader('📊 Correlation Analysis')
            correlation_matrix = filtered_df[
                ['discounted_price', 'rating', 'rating_count', 'real_discount']
            ].corr()
            
            labels = {
                'discounted_price': 'Price',
                'rating': 'Rating',
                'rating_count': 'Reviews',
                'real_discount': 'Discount'
            }
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=list(labels.values()),
                y=list(labels.values()),
                text=correlation_matrix.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title='Correlation Matrix',
                height=400,
                hoverlabel=dict(bgcolor="white"),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader('Price Elasticity Analysis')
            
            # 计算价格弹性系数
            elasticity = elasticity_analyzer.calculate_elasticity(
                filtered_df['discounted_price'].values,
                filtered_df['rating_count'].values,
                method=elasticity_method
            )
            
            # 显示弹性系数及其含义
            st.metric('Price Elasticity', f"{elasticity:.2f}")
            
            if elasticity < 0.5:
                st.success("""
                **低价格弹性** (< 0.5):
                - 消费者对价格变化不敏感
                - 可以考虑适当提高价格
                - 重点关注产品质量和品牌建设
                """)
            else:
                st.warning("""
                **高价格弹性** (≥ 0.5):
                - 消费者对价格变化敏感
                - 需要谨慎调整价格
                - 关注竞品定价策略
                """)
            
            # 根据不同的计算方法显示不同的价格-需求关系图
            if elasticity_method == 'Log-Log':
                # 对数转换后的散点图
                fig = px.scatter(
                    filtered_df,
                    x=np.log(filtered_df['discounted_price']),
                    y=np.log(filtered_df['rating_count']),
                    title='Log-Log Price vs Demand',
                    labels={
                        'x': 'Log Price',
                        'y': 'Log Demand'
                    },
                    trendline="ols"
                )
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
        st.header('📝 Review Analysis')
        
        # 1. 情感分布指标
        total_reviews = len(df)
        positive_count = (df['sentiment'] == 1.0).sum()
        neutral_count = (df['sentiment'] == 0.0).sum()
        negative_count = (df['sentiment'] == -1.0).sum()
        
        positive_ratio = (positive_count / total_reviews) * 100
        neutral_ratio = (neutral_count / total_reviews) * 100
        negative_ratio = (negative_count / total_reviews) * 100
        
        st.markdown("### 评论情感分布")
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.metric(
                "积极评论",
                f"{positive_ratio:.1f}%",
                f"{positive_count} 条评论"
            )
        
        with metric_cols[1]:
            st.metric(
                "中性评论",
                f"{neutral_ratio:.1f}%",
                f"{neutral_count} 条评论"
            )
        
        with metric_cols[2]:
            st.metric(
                "消极评论",
                f"{negative_ratio:.1f}%",
                f"{negative_count} 条评论"
            )
        
        # 2. 情感词分析
        st.markdown("### 情感词分析")
        
        # 获取正面和负面评论的文本
        positive_text = ' '.join(df[df['sentiment'] == 1.0]['review_content'].astype(str))
        negative_text = ' '.join(df[df['sentiment'] == -1.0]['review_content'].astype(str))
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 积极评论关键词")
            # 词云图
            if positive_text:
                fig_pos = create_wordcloud(positive_text, "Positive Reviews", 'positive')
                st.pyplot(fig_pos)
            
            # 积极情感词频率直方图
            top_positive = get_top_sentiment_words(positive_text, 'positive', 10)
            if top_positive:
                # 创建渐变绿色（频次高的颜色更深）
                n_bars = len(top_positive)
                green_colors = [
                    f'rgba(40, {200 - i * 15}, 40, {1 - i * 0.05})'  # 从深到浅的绿色
                    for i in range(n_bars)
                ]
                
                fig_pos_freq = go.Figure()
                fig_pos_freq.add_trace(go.Bar(
                    x=[word for word, _ in top_positive],  # 不需要反转，保持原有顺序（频次从高到低）
                    y=[freq for _, freq in top_positive],
                    marker_color=green_colors,  # 颜色列表从深到浅
                    hovertemplate='词语: %{x}<br>频次: %{y}<extra></extra>'
                ))
                
                fig_pos_freq.update_layout(
                    title="Top 10 Positive Words",
                    xaxis_title="Words",
                    yaxis_title="Frequency",
                    showlegend=False,
                    xaxis_tickangle=-45,
                    height=400,
                    plot_bgcolor='white',
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    margin=dict(l=50, r=20, t=50, b=80)
                )
                st.plotly_chart(fig_pos_freq)
        
        with col2:
            st.markdown("#### 消极评论关键词")
            # 词云图
            if negative_text:
                fig_neg = create_wordcloud(negative_text, "Negative Reviews", 'negative')
                st.pyplot(fig_neg)
            
            # 消极情感词频率直方图
            top_negative = get_top_sentiment_words(negative_text, 'negative', 10)
            if top_negative:
                # 创建渐变红色（频次高的颜色更深）
                n_bars = len(top_negative)
                red_colors = [
                    f'rgba({255 - i * 10}, {20 + i * 5}, {20 + i * 5}, {1 - i * 0.05})'  # 从深到浅的红色
                    for i in range(n_bars)
                ]
                
                fig_neg_freq = go.Figure()
                fig_neg_freq.add_trace(go.Bar(
                    x=[word for word, _ in top_negative],  # 不需要反转，保持原有顺序（频次从高到低）
                    y=[freq for _, freq in top_negative],
                    marker_color=red_colors,  # 颜色列表从深到浅
                    hovertemplate='词语: %{x}<br>频次: %{y}<extra></extra>'
                ))
                
                fig_neg_freq.update_layout(
                    title="Top 10 Negative Words",
                    xaxis_title="Words",
                    yaxis_title="Frequency",
                    showlegend=False,
                    xaxis_tickangle=-45,
                    height=400,
                    plot_bgcolor='white',
                    yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                    margin=dict(l=50, r=20, t=50, b=80)
                )
                st.plotly_chart(fig_neg_freq)
        
        # 3. 评分分布（移到最下面）
        st.markdown("### 评分分布")
        rating_counts = df['rating'].value_counts().sort_index()
        fig_rating = go.Figure()
        
        # 添加柱状图
        fig_rating.add_trace(go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker_color='rgb(0, 123, 255)',
            hovertemplate='评分: %{x}<br>数量: %{y}<extra></extra>'
        ))
        
        # 更新布局
        fig_rating.update_layout(
            title={
                'text': '评分分布',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='评分',
                tickmode='array',
                ticktext=['1', '2', '3', '4', '5'],
                tickvals=[1, 2, 3, 4, 5],
                tickangle=0,
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title='评论数量',
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            plot_bgcolor='white',
            showlegend=False,
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            bargap=0.2
        )
        
        # 添加平均评分标注
        avg_rating = df['rating'].mean()
        fig_rating.add_vline(
            x=avg_rating,
            line_dash="dash",
            line_color="red",
            annotation_text=f"平均评分: {avg_rating:.2f}",
            annotation_position="top"
        )
        
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with tab3:
        st.subheader('Top Rated Products')
        top_products = filtered_df.nlargest(10, 'rating')[
            ['product_name', 'discounted_price', 'rating', 'rating_count']
        ].reset_index(drop=True)
        
        # 使用更好的表格展示
        st.dataframe(
            top_products,
            column_config={
                "product_name": "Product Name",
                "discounted_price": st.column_config.NumberColumn(
                    "Price (₹)",
                    format="₹%.2f"
                ),
                "rating": st.column_config.NumberColumn(
                    "Rating",
                    format="%.1f ⭐"
                ),
                "rating_count": st.column_config.NumberColumn(
                    "Reviews",
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