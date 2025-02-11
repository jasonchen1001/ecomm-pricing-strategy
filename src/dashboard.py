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
    
    # 侧边栏优化
    st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/4/4a/Amazon_icon.svg', width=100)
    st.sidebar.title('Analysis Controls')
    
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
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Price Distribution')
            fig = px.histogram(
                filtered_df,
                x='discounted_price',
                nbins=30,
                title='Price Distribution',
                labels={'discounted_price': 'Price (₹)', 'count': 'Count'},
                hover_data=['discounted_price']
            )
            fig.update_layout(
                showlegend=False,
                hovermode='x',
                hoverlabel=dict(bgcolor="white"),
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader('Price vs Demand')
            elasticity_analyzer = PriceElasticityAnalyzer()
            elasticity = elasticity_analyzer.calculate_elasticity(
                filtered_df['discounted_price'].values,
                filtered_df['rating_count'].values
            )
            st.metric('Price Elasticity', f"{elasticity:.2f}")
            
            fig = px.scatter(
                filtered_df,
                x='discounted_price',
                y='rating_count',
                title='Price vs Demand',
                labels={
                    'discounted_price': 'Price (₹)',
                    'rating_count': 'Demand (Reviews)'
                },
                hover_data=['product_name', 'rating']
            )
            fig.update_layout(
                hovermode='closest',
                hoverlabel=dict(bgcolor="white"),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader('Customer Sentiment Analysis')
        sentiment_analyzer = SentimentAnalyzer()
        filtered_df = sentiment_analyzer.analyze_reviews(filtered_df)
        
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
        
        # 添加词云图
        st.subheader('Common Words in Reviews')
        col1, col2 = st.columns(2)
        with col1:
            st.write("Positive Reviews Word Cloud")
            # 这里可以添加词云图的代码
        with col2:
            st.write("Negative Reviews Word Cloud")
            # 这里可以添加词云图的代码
    
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
    
    # 相关性分析使用 Plotly
    st.header('📊 Correlation Analysis')
    correlation_matrix = filtered_df[
        ['discounted_price', 'rating', 'rating_count', 'real_discount']
    ].corr()
    
    # 使用 Plotly 热力图
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
        height=500,
        hoverlabel=dict(bgcolor="white"),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加页脚
    st.markdown("""---""")
    st.markdown("""
        <div style='text-align: center'>
            <p>Made with ❤️ by Your Team | Data last updated: 2024</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main() 