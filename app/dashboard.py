import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configuration
st.set_page_config(page_title="Advanced Data Dashboard", page_icon="📊", layout="wide")
MAX_ROWS = 10000  # Maximum rows to process
PLOT_CONFIG = {'displayModeBar': False, 'displaylogo': False}
SAMPLE_SIZE = 1000  # For initial display

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(uploaded_file, file_extension):
    """Optimized data loading with caching"""
    try:
        if file_extension == "csv":
            return pd.read_csv(uploaded_file, 
                             engine='c',
                             infer_datetime_format=True,
                             parse_dates=True)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

def create_filter_mask(_df, filters):
    """Vectorized filtering for better performance"""
    if not filters:
        return None
    
    mask = pd.Series(True, index=_df.index)
    for col, value in filters.items():
        if isinstance(value, list):
            if value:
                mask &= _df[col].isin(value)
        else:
            mask &= (_df[col] == value)
    return mask

def create_visualization(fig_func, data, *args, **kwargs):
    """Optimized visualization creation"""
    try:
        fig = fig_func(data, *args, **kwargs)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def display_kpis(df, cols_to_show):
    """Efficient KPI display"""
    if not cols_to_show:
        return
    
    with st.expander("Key Metrics", expanded=True):
        avg_values = df[cols_to_show].mean()
        kpi_cols = st.columns(len(cols_to_show))
        for i, col in enumerate(cols_to_show):
            with kpi_cols[i]:
                st.metric(
                    label=f"Avg {col}", 
                    value=f"{avg_values[col]:,.2f}",
                    help=f"Average value of {col}"
                )

def plot_categorical_distributions(df, categorical_cols):
    """Visualize distributions of categorical variables"""
    if not categorical_cols:
        return
    
    st.subheader("Categorical Variable Distributions")
    cols = st.columns(2)
    
    for i, col in enumerate(categorical_cols):
        fig = px.bar(df[col].value_counts().reset_index(), 
                    x='index', y=col,
                    title=f"Distribution of {col}",
                    labels={'index': col, col: 'Count'})
        cols[i%2].plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

def plot_numerical_relationships(df, numerical_cols):
    """Visualize relationships between numerical variables"""
    if len(numerical_cols) < 2:
        return
    
    st.subheader("Numerical Variable Relationships")
    
    # Correlation matrix
    corr_matrix = df[numerical_cols].corr()
    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   title="Correlation Matrix",
                   color_continuous_scale='RdBu',
                   range_color=[-1, 1])
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    
    # Pair plot for top 5 numerical variables
    if len(numerical_cols) > 2:
        num_vars = numerical_cols[:5]  # Limit to 5 variables for performance
        fig = px.scatter_matrix(df[num_vars],
                               title="Pair Plot of Numerical Variables",
                               height=800)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

def plot_mixed_relationships(df, numerical_cols, categorical_cols):
    """Visualize relationships between numerical and categorical variables"""
    if not numerical_cols or not categorical_cols:
        return
    
    st.subheader("Numerical vs Categorical Relationships")
    
    # For each categorical variable, show distribution of numerical variables
    for cat_col in categorical_cols[:3]:  # Limit to 3 categorical variables
        st.write(f"### Relationship with {cat_col}")
        cols = st.columns(2)
        
        # Box plots for each numerical variable
        for i, num_col in enumerate(numerical_cols[:4]):  # Limit to 4 numerical
            fig = px.box(df, x=cat_col, y=num_col,
                        title=f"{num_col} by {cat_col}")
            cols[i%2].plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

def main():
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV or Excel)", 
        type=["csv", "xlsx"],
        help="For best performance, use CSV files under 10MB"
    )
    
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        df = load_data(uploaded_file, file_extension)
        
        if df.empty:
            st.warning("The uploaded file is empty or couldn't be loaded.")
            return
            
        display_df = df.sample(min(SAMPLE_SIZE, len(df))) if len(df) > MAX_ROWS else df
        if len(df) > MAX_ROWS:
            st.warning(f"Large dataset detected ({len(df):,} rows). Showing sample of {SAMPLE_SIZE} rows for faster display.")

        with st.expander("View Raw Data", expanded=False):
            page_size = st.slider('Rows per page', 10, 100, 20, key='pagination')
            page_num = st.number_input('Page', 1, max(1, (len(display_df)//page_size)+1), 1)
            st.dataframe(display_df.iloc[(page_num-1)*page_size : page_num*page_size])

        # Identify column types
        categorical_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
        numerical_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        # Sidebar filters
        filters = {}
        if categorical_cols:
            st.sidebar.header("Filter Data")
            for col in categorical_cols:
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 10:
                    filters[col] = st.sidebar.radio(
                        f"Select {col}", 
                        options=unique_values,
                        key=f"filter_{col}"
                    )
                else:
                    filters[col] = st.sidebar.multiselect(
                        f"Select {col}", 
                        options=unique_values,
                        default=unique_values[:5] if len(unique_values) > 5 else unique_values,
                        key=f"filter_{col}"
                    )

        filter_mask = create_filter_mask(df, filters)
        df_filtered = df[filter_mask] if filter_mask is not None else df.copy()
        
        if df_filtered.empty:
            st.warning("No data available based on the current filters.")
            return

        st.title("📊 Advanced Data Dashboard")
        
        if numerical_cols:
            visible_metrics = numerical_cols[:min(3, len(numerical_cols))]
            display_kpis(df_filtered, visible_metrics)
            st.divider()

        # New relationship visualizations
        plot_categorical_distributions(df_filtered, categorical_cols)
        plot_numerical_relationships(df_filtered, numerical_cols)
        plot_mixed_relationships(df_filtered, numerical_cols, categorical_cols)

        # Original visualizations
        if len(numerical_cols) >= 2:
            st.header("Detailed Visualizations")
            
            chart_options = st.multiselect(
                "Select detailed charts to display",
                options=["Histogram", "Scatter Plot", "Box Plot", "Bar Chart", "Line Chart", "Pie Chart"],
                default=["Histogram", "Scatter Plot"],
                key="chart_selector"
            )
            
            with ThreadPoolExecutor() as executor:
                figures = {}
                num_col1, num_col2 = numerical_cols[:2]
                
                if "Histogram" in chart_options:
                    figures['hist'] = executor.submit(
                        create_visualization,
                        px.histogram,
                        df_filtered,
                        x=num_col1,
                        title=f"Distribution of {num_col1}",
                        render_mode='webgl'
                    )
                
                if "Scatter Plot" in chart_options:
                    figures['scatter'] = executor.submit(
                        create_visualization,
                        px.scatter,
                        df_filtered,
                        x=num_col1,
                        y=num_col2,
                        title=f"{num_col1} vs {num_col2}",
                        render_mode='webgl'
                    )
                
                if "Box Plot" in chart_options:
                    figures['box'] = executor.submit(
                        create_visualization,
                        px.box,
                        df_filtered,
                        y=num_col1,
                        title=f"Distribution of {num_col1}"
                    )
                
                if "Bar Chart" in chart_options:
                    figures['bar'] = executor.submit(
                        create_visualization,
                        px.bar,
                        df_filtered,
                        x=num_col1,
                        y=num_col2,
                        title=f"{num_col1} vs {num_col2}"
                    )
                
                if "Line Chart" in chart_options:
                    figures['line'] = executor.submit(
                        create_visualization,
                        px.line,
                        df_filtered,
                        x=num_col1,
                        y=num_col2,
                        title=f"{num_col1} vs {num_col2}"
                    )
                
                if "Pie Chart" in chart_options and categorical_cols:
                    figures['pie'] = executor.submit(
                        create_visualization,
                        px.pie,
                        df_filtered,
                        names=categorical_cols[0],
                        title=f"Distribution of {categorical_cols[0]}"
                    )
                
                cols = st.columns(2)
                col_index = 0
                
                for fig_name, future in figures.items():
                    fig = future.result()
                    if fig:
                        with cols[col_index % 2]:
                            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                        col_index += 1

        with st.expander("View Filtered Data", expanded=False):
            st.dataframe(df_filtered.head(100))

if __name__ == "__main__":
    main()
