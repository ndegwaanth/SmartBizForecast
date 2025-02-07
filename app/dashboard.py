import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dynamic Data Dashboard", page_icon="📊", layout="wide")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display raw data
    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Sidebar filters for categorical variables
    if categorical_cols:
        st.sidebar.header("Filter Data")
        filters = {}

        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 10:
                filters[col] = st.sidebar.radio(f"Select {col}", options=unique_values)
            else:
                filters[col] = st.sidebar.multiselect(f"Select {col}", options=unique_values, default=unique_values)

        # Apply filters
        df_filtered = df.copy()
        for col, value in filters.items():
            if isinstance(value, list):
                df_filtered = df_filtered[df_filtered[col].isin(value)]
            else:
                df_filtered = df_filtered[df_filtered[col] == value]
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.warning("No data available based on the current filters.")
        st.stop()

    # KPIs
    st.title("📊 Dynamic Data Dashboard")

    if numerical_cols:
        avg_values = df_filtered[numerical_cols].mean()
        min_values = df_filtered[numerical_cols].min()
        max_values = df_filtered[numerical_cols].max()

        kpi_cols = st.columns(min(3, len(numerical_cols)))
        for i, col in enumerate(numerical_cols[:3]):
            with kpi_cols[i]:
                st.metric(label=f"Avg {col}", value=f"{avg_values[col]:,.2f}")

        st.divider()

    # Visualizations
    if len(numerical_cols) >= 2:
        num_col1, num_col2 = numerical_cols[:2]

        # Histogram
        fig1 = px.histogram(df_filtered, x=num_col1, title=f"Distribution of {num_col1}")

        # Scatter Plot
        fig2 = px.scatter(df_filtered, x=num_col1, y=num_col2, title=f"Scatter Plot: {num_col1} vs {num_col2}")

        # Box Plot
        fig3 = px.box(df_filtered, y=num_col1, title=f"Box Plot: {num_col1}")

        # Bar Chart
        fig4 = px.bar(df_filtered, x=num_col1, y=num_col2, title=f"Bar Chart: {num_col1} vs {num_col2}")

        # Line Chart
        fig5 = px.line(df_filtered, x=num_col1, y=num_col2, title=f"Line Chart: {num_col1} vs {num_col2}")

        # Pie Chart (only if categorical data exists)
        if categorical_cols:
            cat_col = categorical_cols[0]
            fig6 = px.pie(df_filtered, names=cat_col, title=f"Pie Chart: Distribution of {cat_col}")

        # Display visuals in a grid
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        col3.plotly_chart(fig3, use_container_width=True)
        col4.plotly_chart(fig4, use_container_width=True)

        col5, col6 = st.columns(2)
        col5.plotly_chart(fig5, use_container_width=True)

        if categorical_cols:
            col6.plotly_chart(fig6, use_container_width=True)

    st.subheader("Filtered Data")
    st.dataframe(df_filtered)

else:
    st.warning("Please upload a dataset to proceed.")