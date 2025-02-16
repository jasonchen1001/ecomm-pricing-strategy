import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# 设置自定义配色方案
COLOR_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD',
    '#D4A5A5', '#9B5DE5', '#F15BB5', '#00BBF9', '#00F5D4'
]

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
root_dir = os.path.dirname(current_dir)

# 页面配置
st.set_page_config(
    page_title="Amazon Pricing Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #f6f8fa, #ffffff);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    .stPlotlyChart {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(root_dir, 'data', 'processed_amazon.csv'))
        recommendations = pd.read_csv(os.path.join(root_dir, 'data', 'price_recommendations.csv'))
        return df, recommendations
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

df, recommendations = load_data()

if df is not None and recommendations is not None:
    # 侧边栏过滤器
    with st.sidebar:
        st.title('🎯 Interactive Filters')
        
        # 类别过滤器
        categories = ['All'] + sorted(df['main_category'].unique().tolist())
        selected_category = st.selectbox(
            '🏷️ Category',
            categories
        )
        
        # 价格范围过滤器
        price_range = st.slider(
            '💰 Price Range (₹)',
            min_value=float(df['discounted_price'].min()),
            max_value=float(df['discounted_price'].max()),
            value=(float(df['discounted_price'].min()), float(df['discounted_price'].max()))
        )
        
        # 评分范围过滤器
        rating_range = st.slider(
            '⭐ Rating Range',
            min_value=1.0,
            max_value=5.0,
            value=(3.0, 5.0),
            step=0.5
        )
        
        # 排序方式
        sort_by = st.radio(
            "📊 Sort Products By",
            ["Price", "Rating", "Discount", "Reviews"]
        )

    # 应用过滤器
    if selected_category != 'All':
        filtered_df = df[df['main_category'] == selected_category]
    else:
        filtered_df = df.copy()
        
    filtered_df = filtered_df[
        (filtered_df['discounted_price'] >= price_range[0]) &
        (filtered_df['discounted_price'] <= price_range[1]) &
        (filtered_df['rating'] >= rating_range[0]) &
        (filtered_df['rating'] <= rating_range[1])
    ]
    
    # 排序
    if sort_by == "Price":
        filtered_df = filtered_df.sort_values('discounted_price', ascending=False)
    elif sort_by == "Rating":
        filtered_df = filtered_df.sort_values('rating', ascending=False)
    elif sort_by == "Discount":
        filtered_df = filtered_df.sort_values('real_discount', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('rating_count', ascending=False)
    
    # 主要内容
    st.title("🚀 Amazon Product Pricing Strategy Analysis")
    
    # 显示过滤后的数据统计
    st.subheader("📊 Key Performance Indicators")
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("📦 Products", f"{len(filtered_df):,}",
                 delta=f"{len(filtered_df)/len(df)*100:.1f}% of total")
    with metrics_cols[1]:
        st.metric("💰 Avg Price", f"₹{filtered_df['discounted_price'].mean():,.2f}",
                 delta=f"₹{filtered_df['discounted_price'].mean() - df['discounted_price'].mean():,.2f}")
    with metrics_cols[2]:
        st.metric("⭐ Avg Rating", f"{filtered_df['rating'].mean():.2f}",
                 delta=f"{filtered_df['rating'].mean() - df['rating'].mean():.2f}")
    with metrics_cols[3]:
        st.metric("🏷️ Avg Discount", f"{filtered_df['real_discount'].mean():.1f}%",
                 delta=f"{filtered_df['real_discount'].mean() - df['real_discount'].mean():.1f}%")
    
    # 创建交互式图表
    tabs = st.tabs(["📈 Price Analysis", "🎯 Market Insights", "💡 Recommendations"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # 价格分布直方图（带滑动选择）
            fig = px.histogram(filtered_df, x="discounted_price", nbins=50,
                             title="Price Distribution",
                             color_discrete_sequence=COLOR_PALETTE)
            fig.update_layout(
                dragmode='select',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 交互式散点图
            fig = px.scatter(filtered_df, 
                           x="real_discount", 
                           y="rating",
                           color="main_category",
                           size="rating_count",
                           hover_data=["product_name", "discounted_price"],
                           title="Discount vs Rating Analysis",
                           color_discrete_sequence=COLOR_PALETTE)
            fig.update_layout(
                dragmode='zoom',
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 价格区间箱线图
            fig = px.box(filtered_df, 
                        x="main_category", 
                        y="discounted_price",
                        color="main_category",
                        title="Price Distribution by Category",
                        color_discrete_sequence=COLOR_PALETTE)
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 热力图
            price_bins = pd.qcut(filtered_df['discounted_price'], 10).astype(str)
            rating_bins = pd.qcut(filtered_df['rating'], 5).astype(str)
            rating_dist = pd.crosstab(price_bins, rating_bins)
            
            fig = px.imshow(rating_dist,
                          title="Price vs Rating Heatmap",
                          color_continuous_scale="Viridis",
                          aspect="auto")
            fig.update_traces(
                hoverongaps=False,
                hovertemplate="Price: %{y}<br>Rating: %{x}<br>Count: %{z}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # 交互式气泡图
            fig = px.scatter(filtered_df,
                           x="discounted_price",
                           y="rating_count",
                           size="real_discount",
                           color="main_category",
                           hover_name="product_name",
                           title="Price vs Popularity Analysis",
                           color_discrete_sequence=COLOR_PALETTE)
            st.plotly_chart(fig, use_container_width=True)
            
            # 堆叠面积图
            price_trends = filtered_df.groupby(['main_category', 'rating'])['discounted_price'].mean().reset_index()
            fig = px.area(price_trends, 
                         x="rating", 
                         y="discounted_price",
                         color="main_category",
                         title="Price Trends by Rating",
                         color_discrete_sequence=COLOR_PALETTE)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 3D散点图
            fig = go.Figure(data=[go.Scatter3d(
                x=filtered_df['discounted_price'],
                y=filtered_df['rating'],
                z=filtered_df['rating_count'],
                mode='markers',
                marker=dict(
                    size=filtered_df['real_discount']/5,
                    color=filtered_df['real_discount'],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=filtered_df['product_name'],
                hovertemplate="Price: ₹%{x:.2f}<br>Rating: %{y:.1f}<br>Reviews: %{z}<br>%{text}<extra></extra>"
            )])
            fig.update_layout(
                title="3D Product Analysis",
                scene=dict(
                    xaxis_title="Price",
                    yaxis_title="Rating",
                    zaxis_title="Reviews"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 新增：类别性能雷达图
            categories = filtered_df['main_category'].unique()
            metrics = ['Rating', 'Reviews', 'Discount', 'Price']  # 更好的标签名称
            
            # 计算每个类别的平均指标
            category_metrics = []
            for category in categories:
                category_data = filtered_df[filtered_df['main_category'] == category]
                metrics_values = []
                for metric, col in zip(metrics, ['rating', 'rating_count', 'real_discount', 'discounted_price']):
                    value = (category_data[col].mean() - filtered_df[col].min()) / \
                            (filtered_df[col].max() - filtered_df[col].min())
                    metrics_values.append(value)
                category_metrics.append(metrics_values)
            
            # 创建雷达图
            fig = go.Figure()

            # 添加背景网格
            fig.add_trace(go.Scatterpolar(
                r=[1, 1, 1, 1],
                theta=metrics,
                fill='toself',
                name='',
                fillcolor='rgba(200, 200, 200, 0.1)',
                line=dict(color='rgba(200, 200, 200, 0.2)'),
                showlegend=False
            ))

            # 添加类别数据
            for i, category in enumerate(categories):
                # 将颜色代码转换为 RGBA
                hex_color = COLOR_PALETTE[i % len(COLOR_PALETTE)].lstrip("#")
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                rgba_color = f'rgba({r}, {g}, {b}, 0.2)'
                
                fig.add_trace(go.Scatterpolar(
                    r=category_metrics[i],
                    theta=metrics,
                    fill='toself',
                    name=category,
                    line=dict(
                        color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                        width=2
                    ),
                    fillcolor=rgba_color
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        showline=False,
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        gridwidth=0.5,
                        tickfont=dict(size=10),
                        ticktext=['0%', '25%', '50%', '75%', '100%'],
                        tickvals=[0, 0.25, 0.5, 0.75, 1]
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(200, 200, 200, 0.3)',
                        gridwidth=0.5,
                        linewidth=0.5,
                        tickfont=dict(size=12, color='#666'),
                        rotation=90,  # 旋转角度
                        direction='clockwise'
                    ),
                    bgcolor='white'
                ),
                title={
                    'text': "Category Performance Analysis",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16)
                },
                showlegend=True,
                height=400,
                width=None,
                margin=dict(t=40, b=40, l=40, r=40),
                paper_bgcolor='white',
                plot_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.95,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(200, 200, 200, 0.5)',
                    borderwidth=1,
                    font=dict(size=10)
                )
            )

            # 使用container包装图表，并设置固定宽度
            with st.container():
                col1, col2, col3 = st.columns([1, 6, 1])
                with col2:
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # 价格建议分析
        if selected_category != 'All':
            filtered_recommendations = recommendations[
                recommendations['product_id'].isin(filtered_df['product_id'])
            ]
        else:
            filtered_recommendations = recommendations
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 瀑布图：价格调整
            fig = go.Figure(go.Waterfall(
                name="Price Changes",
                orientation="v",
                measure=["relative"] * len(filtered_recommendations),
                x=list(filtered_recommendations.index),
                y=filtered_recommendations['adjusted_change'],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
                decreasing={"marker":{"color":"#FF6B6B"}},
                increasing={"marker":{"color":"#4ECDC4"}},
                text=filtered_recommendations['adjusted_change'].round(1).astype(str) + '%',
                textposition="outside"
            ))
            fig.update_layout(
                title="Price Adjustment Distribution",
                showlegend=False,
                xaxis_title="Products",
                yaxis_title="Price Change (%)",
                height=400,
                margin=dict(t=30, b=0, l=0, r=0)  # 减小边距
            )
            fig.update_xaxes(
                ticktext=filtered_recommendations['product_id'],
                tickvals=list(filtered_recommendations.index),
                tickmode='array',
                tickangle=45,
                showticklabels=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 收入影响分析
            revenue_change = ((filtered_recommendations['expected_revenue'].sum() - 
                              filtered_recommendations['current_revenue'].sum()) / 
                             filtered_recommendations['current_revenue'].sum() * 100)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current Revenue',
                x=['Revenue'],
                y=[filtered_recommendations['current_revenue'].sum()],
                marker_color='#FF6B6B',
                width=0.3  # 减小柱子宽度
            ))
            fig.add_trace(go.Bar(
                name='Expected Revenue',
                x=['Revenue'],
                y=[filtered_recommendations['expected_revenue'].sum()],
                marker_color='#4ECDC4',
                width=0.3
            ))
            fig.update_layout(
                title=f'Revenue Impact (Expected Growth: {revenue_change:.1f}%)',
                barmode='group',
                height=400,
                margin=dict(t=30, b=0, l=0, r=0),
                bargap=0.15  # 调整柱子间距
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 产品表格
        st.subheader("🔍 Key Products Analysis")
        top_tabs = st.tabs(["📈 Top Price Increases", "📉 Top Price Decreases"])

        with top_tabs[0]:
            top_increases = filtered_recommendations.nlargest(5, 'adjusted_change')
            st.dataframe(
                top_increases[['product_id', 'current_price', 'recommended_price', 
                              'adjusted_change', 'expected_revenue']].style\
                    .format({
                        'current_price': '₹{:.2f}',
                        'recommended_price': '₹{:.2f}',
                        'adjusted_change': '{:.1f}%',
                        'expected_revenue': '₹{:.2f}'
                    })\
                    .background_gradient(subset=['adjusted_change'], cmap='RdYlGn')\
                    .set_properties(**{'text-align': 'center'})
            )

        with top_tabs[1]:
            top_decreases = filtered_recommendations.nsmallest(5, 'adjusted_change')
            st.dataframe(
                top_decreases[['product_id', 'current_price', 'recommended_price', 
                              'adjusted_change', 'expected_revenue']].style\
                    .format({
                        'current_price': '₹{:.2f}',
                        'recommended_price': '₹{:.2f}',
                        'adjusted_change': '{:.1f}%',
                        'expected_revenue': '₹{:.2f}'
                    })\
                    .background_gradient(subset=['adjusted_change'], cmap='RdYlGn_r')\
                    .set_properties(**{'text-align': 'center'})
            )

    # 页脚
    st.markdown("---")
    st.markdown("*Dashboard by Yanzhen Chen / 陈彦臻*")
else:
    st.error("无法加载数据。请确保数据文件存在且格式正确。")