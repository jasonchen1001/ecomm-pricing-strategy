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

def main():
    # 标题和介绍
    st.title('📊 Amazon Cable Products Pricing Analysis')
    st.markdown("""
    This dashboard provides comprehensive analysis of cable products pricing on Amazon India.
    Use the filters in the sidebar to explore different price ranges and product categories.
    """)
    
    # 加载数据时显示进度条
    with st.spinner('Loading data...'):
        df = load_data('amazon.csv')
        features = extract_features(df)
    
    # 创建情感分析器实例并分析评论
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_reviews(df)  # 添加 sentiment 列
    
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
        
        # 计算情感统计
        filtered_df = df.copy()  # 使用包含 sentiment 列的数据
        
        col1, col2 = st.columns(2)
        
        with col1:
            positive_ratio = (filtered_df['sentiment'] > 0).mean() * 100
            st.metric(
                'Positive Reviews',
                f"{positive_ratio:.1f}%",
                delta=f"{positive_ratio - 50:.1f}% from neutral"
            )
        
        with col2:
            negative_ratio = (filtered_df['sentiment'] < 0).mean() * 100
            st.metric(
                'Negative Reviews',
                f"{negative_ratio:.1f}%",
                delta=f"{negative_ratio - 50:.1f}% from neutral",
                delta_color="inverse"
            )
        
        # 添加词云分析
        st.header("📊 评论词云分析")
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        # 生成积极评论词云
        with col1:
            st.subheader("积极评论词云")
            positive_reviews = filtered_df[filtered_df['sentiment'] > 0]['review_content'].fillna('').str.cat(sep=' ')
            if positive_reviews:
                # 生成词云
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='YlGn'  # 使用绿色系配色
                ).generate(positive_reviews)
                
                # 显示词云图
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        
        # 生成消极评论词云
        with col2:
            st.subheader("消极评论词云")
            negative_reviews = filtered_df[filtered_df['sentiment'] < 0]['review_content'].fillna('').str.cat(sep=' ')
            if negative_reviews:
                # 生成词云
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    max_words=100,
                    colormap='RdPu'  # 使用红色系配色
                ).generate(negative_reviews)
                
                # 显示词云图
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        
        # 显示高频词统计
        st.subheader("📈 高频词统计")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 积极评论高频词")
            positive_freq = pd.Series(dict(sentiment_analyzer.get_frequent_words(
                df[df['sentiment'] > 0]['review_content'],
                sentiment_type='positive'
            )))
            
            # 创建积极评论高频词柱状图
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=positive_freq.head(10).index,
                y=positive_freq.head(10).values,
                marker_color='rgb(50, 205, 50)',  # 设置为绿色
                marker_line_color='rgb(25, 102, 25)',
                marker_line_width=1,
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Top 10 Words in Positive Reviews',
                plot_bgcolor='white',
                bargap=0.3,
                showlegend=False,
                xaxis=dict(
                    title='Words',
                    gridcolor='lightgrey',
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                yaxis=dict(
                    title='Frequency',
                    gridcolor='lightgrey',
                    showgrid=True,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col4:
            st.markdown("### 消极评论高频词")
            negative_freq = pd.Series(dict(sentiment_analyzer.get_frequent_words(
                df[df['sentiment'] < 0]['review_content'],
                sentiment_type='negative'
            )))
            
            # 创建消极评论高频词柱状图
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=negative_freq.head(10).index,
                y=negative_freq.head(10).values,
                marker_color='rgb(255, 99, 71)',  # 设置为红色
                marker_line_color='rgb(139, 26, 26)',
                marker_line_width=1,
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Top 10 Words in Negative Reviews',
                plot_bgcolor='white',
                bargap=0.3,
                showlegend=False,
                xaxis=dict(
                    title='Words',
                    gridcolor='lightgrey',
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                yaxis=dict(
                    title='Frequency',
                    gridcolor='lightgrey',
                    showgrid=True,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
    
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

if __name__ == '__main__':
    main() 