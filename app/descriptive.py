import pandas as pd
import numpy as np
import scipy.stats as stats

class Descriptive:
    @staticmethod
    def generate_descriptive_stats(df):
        """
        Generates descriptive and inferential statistics for any dataset.
        Parameters:
            df (pd.DataFrame): The input dataset.
        Returns:
            dict: A dictionary with structured statistics.
        """
        stats_dict = {}

        # Basic Descriptive Statistics
        stats_dict['Summary Statistics'] = df.describe(include='all').reset_index()

        # Missing values
        stats_dict['Missing Values'] = pd.DataFrame(df.isnull().sum(), columns=['Missing Count']).reset_index()

        # Data types
        stats_dict['Data Types'] = pd.DataFrame(df.dtypes, columns=['Type']).reset_index()

        # Skewness & Kurtosis for numerical columns
        num_cols = df.select_dtypes(include=['number'])
        stats_dict['Skewness'] = pd.DataFrame(num_cols.skew(), columns=['Skewness']).reset_index()
        stats_dict['Kurtosis'] = pd.DataFrame(num_cols.kurtosis(), columns=['Kurtosis']).reset_index()

        # Correlation Matrix
        stats_dict['Correlation Matrix'] = num_cols.corr()

        # Unique values for categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category'])
        stats_dict['Unique Values'] = pd.DataFrame({col: [df[col].nunique()] for col in cat_cols}).T
        stats_dict['Unique Values'].columns = ['Unique Count']

        # Confidence Intervals for numerical columns
        ci_list = []
        for col in num_cols:
            mean = np.mean(df[col].dropna())
            std_err = stats.sem(df[col].dropna())
            ci = stats.t.interval(0.95, len(df[col].dropna()) - 1, loc=mean, scale=std_err)
            ci_list.append([col, ci[0], ci[1]])
        stats_dict['Confidence Intervals'] = pd.DataFrame(ci_list, columns=['Feature', 'Lower Bound', 'Upper Bound'])

        # Normality Tests (Shapiro-Wilk Test)
        normality_tests = []
        for col in num_cols:
            stat, p = stats.shapiro(df[col].dropna())
            normality_tests.append([col, stat, p])
        stats_dict['Normality Tests'] = pd.DataFrame(normality_tests, columns=['Feature', 'Shapiro-Wilk Statistic', 'P-Value'])

        # Regression Analysis (Simple Linear Regression Coefficients for first two numerical columns if available)
        if len(num_cols.columns) >= 2:
            x_col, y_col = num_cols.columns[:2]
            x_values = df[x_col].dropna()
            y_values = df[y_col].dropna()

            # Ensure x values are not all identical
            if x_values.nunique() > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                stats_dict['regression_analysis'] = {
                    'slope': slope, 'intercept': intercept, 'r_squared': r_value**2, 'p_value': p_value, 'std_err': std_err
                }
            else:
                stats_dict['regression_analysis'] = "Linear regression cannot be computed because all x values are identical."


        return stats_dict