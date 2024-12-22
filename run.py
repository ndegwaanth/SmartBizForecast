import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data():
  df = pd.read_csv('data/New-Online_Retail.csv')
  return df

df = load_data()

st.sidebar.header("Filter the columns")

# filtering with the country
country = st.sidebar.multiselect(
  "Select the Country",
  options = df['Country'].unique(),
  default = df['Country'].unique()
)

# Automation sidebar
customer_id = st.sidebar.radio(
  "Select the customer ID",
  options = df['CustomerID'].unique(),
)

df_select = df.query(
  "Country == @country & CustomerID == @customer_id"
)

st.dataframe(df_select)