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
    
    # 在侧边栏添加价格弹性分析设置
    st.sidebar.markdown("---")
    st.sidebar.subheader("Price Elasticity Settings")
    
    # 弹性阈值设置
    elasticity_threshold = st.sidebar.slider(
        'Elasticity Threshold',
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Threshold to determine high/low price elasticity"
    )
    
    # 添加价格区间选择
    price_segments = st.sidebar.number_input(
        'Price Segments',
        min_value=2,
        max_value=10,
        value=5,
        help="Number of price segments for elasticity analysis"
    )
    
    # 添加计算方法选择
    elasticity_method = st.sidebar.selectbox(
        'Calculation Method',
        ['Log-Log', 'Point', 'Arc'],
        help="Method to calculate price elasticity"
    )
    
    # 添加时间窗口选择（如果有时间序列数据）
    time_window = st.sidebar.selectbox(
        'Analysis Period',
        ['All Time', 'Last 30 Days', 'Last 90 Days', 'Last 180 Days'],
        help="Time period for elasticity analysis"
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
            st.subheader('Price Elasticity Analysis')
            elasticity_analyzer = PriceElasticityAnalyzer()
            elasticity = elasticity_analyzer.calculate_elasticity(
                filtered_df['discounted_price'].values,
                filtered_df['rating_count'].values,
                method=elasticity_method,
                segments=price_segments
            )
            
            # 显示分析结果
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric('Price Elasticity', f"{elasticity:.2f}")
            with metrics_col2:
                st.metric('Threshold', f"{elasticity_threshold:.2f}")
            with metrics_col3:
                st.metric('Method', elasticity_method)
            
            # 根据用户设置的阈值判断
            if elasticity < elasticity_threshold:
                st.info(f"""
                💡 低弹性市场特征 (< {elasticity_threshold:.2f}):
                - 消费者对价格不敏感
                - 具有较强的定价能力
                - 建议策略：
                  1. 可以适当提高价格
                  2. 重点关注产品质量和品牌建设
                  3. 通过差异化竞争而非价格战
                """)
            else:
                st.warning(f"""
                ⚠️ 高弹性市场特征 (≥ {elasticity_threshold:.2f}):
                - 消费者对价格敏感
                - 价格竞争激烈
                - 建议策略：
                  1. 保持价格竞争力
                  2. 密切关注竞品定价
                  3. 考虑促销策略
                """)
            
            # Price vs Demand 图表部分
            fig = go.Figure()
            
            # 添加基础散点图
            fig.add_trace(go.Scatter(
                x=filtered_df['discounted_price'],
                y=filtered_df['rating_count'],
                mode='markers',
                name='Raw Data',
                marker=dict(
                    color='gray',
                    opacity=0.5,
                    size=8
                ),
                hovertemplate='<br>'.join([
                    'Price: ₹%{x:.2f}',
                    'Demand: %{y}',
                    '<extra></extra>'
                ])
            ))
            
            # 根据不同方法添加趋势线
            if elasticity_method == 'Log-Log':
                # 对数回归拟合线
                mask = (filtered_df['discounted_price'] > 0) & (filtered_df['rating_count'] > 0)
                prices_clean = filtered_df['discounted_price'][mask]
                quantities_clean = filtered_df['rating_count'][mask]
                
                X = np.log(prices_clean)
                y = np.log(quantities_clean)
                model = LinearRegression()
                model.fit(X.values.reshape(-1, 1), y)
                
                # 生成预测线
                x_range = np.linspace(X.min(), X.max(), 100)
                y_pred = model.predict(x_range.reshape(-1, 1))
                
                fig.add_trace(go.Scatter(
                    x=np.exp(x_range),
                    y=np.exp(y_pred),
                    mode='lines',
                    name=f'Log-Log Fit (e={elasticity:.2f})',
                    line=dict(color='red', width=2)
                ))
                
            else:  # Point 或 Arc 方法
                # 计算分段平均值
                prices_series = pd.Series(filtered_df['discounted_price'])
                quantities_series = pd.Series(filtered_df['rating_count'])
                price_bins = pd.qcut(prices_series, price_segments)
                avg_quantities = quantities_series.groupby(price_bins).mean()
                avg_prices = prices_series.groupby(price_bins).mean()
                
                # 添加分段线和点
                fig.add_trace(go.Scatter(
                    x=avg_prices,
                    y=avg_quantities,
                    mode='lines+markers',
                    name=f'{elasticity_method} Segments',
                    line=dict(color='red', width=2),
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='circle'
                    ),
                    hovertemplate='<br>'.join([
                        'Segment Average:',
                        'Price: ₹%{x:.2f}',
                        'Demand: %{y:.0f}',
                        '<extra></extra>'
                    ])
                ))
                
                # 添加段号标签
                for i, (p, q) in enumerate(zip(avg_prices, avg_quantities)):
                    fig.add_annotation(
                        x=p,
                        y=q,
                        text=f'Segment {i+1}',
                        showarrow=True,
                        arrowhead=1
                    )
            
            # 更新布局
            fig.update_layout(
                title=f'Price vs Demand ({elasticity_method} Method)',
                xaxis_title='Price (₹)',
                yaxis_title='Demand (Reviews)',
                hovermode='closest',
                showlegend=True,
                height=500,
                template='plotly_white',
                hoverlabel=dict(bgcolor="white"),
                margin=dict(t=50, l=50, r=50, b=50)
            )
            
            # 添加网格线
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
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
            <p>Made with ❤️ by Yanzhen Chen | Data last updated: 2025</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main() 