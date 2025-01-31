import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Automating Missing values
def hadle_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ['int64', 'float4']: 
            '''Numerical colums'''
            df[column].fillna(df[column].mean(), inplace=True)
        elif df[column].dtype == 'object':
            """Categorical variable"""
            df[column].fillna(df[column].mode()[0], inplace=True)
    return df


# Removing Duplicates
def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df


# Handling Outliers using interquartile range
def remove_outliers(df):
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


# Standardize The data types of the dataset
def standardized_data_types(df):
    for column in df.columns:
        if 'date' in column.lower():
            """Columns with 'date' in their names """
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif df[column].dtype == 'object':
            """ Converting object to category if there is """
            df[column] = df[column].astype('category')
    return df


# Normalize or scaling the Numerical Data
def scale_numeric_data(df):
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


# Overal data cleaning functions in a single function
def data_cleaning(df):
    # Handling Missing Values
    df = hadle_missing_values(df)

    # Removing Duplicates
    df = remove_duplicates(df)

    # Removing Outliers
    df = remove_outliers(df)

    # standardize data types
    df = standardized_data_types(df)

    # Scaling Numerical values
    df = scale_numeric_data(df)
