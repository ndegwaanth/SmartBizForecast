import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataCleaning:

    # Automating Missing Values
    def handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.
        - For numerical columns: Fill with the mean.
        - For categorical columns: Fill with the mode.
        """
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                if df[column].isnull().any():
                    df[column].fillna(df[column].mean(), inplace=True)
            elif df[column].dtype == 'object':
                if df[column].isnull().any():
                    df[column].fillna(df[column].mode()[0], inplace=True)
        return df


    # Removing Duplicates
    def remove_duplicates(self, df):
        """
        Remove duplicate rows from the DataFrame.
        """
        if df.duplicated().any(): 
            df.drop_duplicates(inplace=True)
        return df


    # Handling Outliers using Interquartile Range (IQR)
    def remove_outliers(self, df):
        """
        Remove outliers from numerical columns using the IQR method.
        """
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_columns) > 0:
            for column in numerical_columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df


    # Standardize Data Types
    def standardize_data_types(self, df):
        """
        Standardize data types in the DataFrame.
        - Convert columns with 'date' in their names to datetime.
        - Convert object columns to category.
        """
        for column in df.columns:
            if 'date' in column.lower():
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                except Exception as e:
                    print(f"Error converting {column} to datetime: {e}")
            elif df[column].dtype == 'object':
                df[column] = df[column].astype('category')
        return df


    # Normalize or Scale Numerical Data
    def scale_numeric_data(self, df):
        """
        Scale numerical data using MinMaxScaler.
        """
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_columns) > 0:
            scaler = MinMaxScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        return df


data = DataCleaning()

# Overall Data Cleaning Function
def data_cleaning(df):
    """
    Perform all data cleaning steps on the DataFrame.
    - Handle missing values.
    - Remove duplicates.
    - Remove outliers.
    - Standardize data types.
    - Scale numerical data.
    """
    print("Original data shape:", df.shape)

    # Handling Missing Values
    df = data.handle_missing_values(df)
    print("After handling missing values:", df.shape)

    # Removing Duplicates
    df = data.remove_duplicates(df)
    print("After removing duplicates:", df.shape)

    # Removing Outliers
    df = data.remove_outliers(df)
    print("After removing outliers:", df.shape)

    # Standardize Data Types
    df = data.standardize_data_types(df)
    print("After standardizing data types:", df.shape)

    # Scaling Numerical Data
    df = data.scale_numeric_data(df)
    print("After scaling numerical data:", df.shape)

    return df