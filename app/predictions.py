import matplotlib
matplotlib.use('Agg')

import os
import base64
import pickle
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from functools import wraps
import signal
import logging



class ChurnPrediction:
    def __init__(self):
        # Initialize the model, scaler, and encoders from scratch
        self.model = SGDClassifier(loss="log_loss", random_state=42)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.classes = None
        self.feature_names = None
    def preprocess_data(self, df, target_column):
        """Prepares data by handling missing values and encoding categorical features."""
        
        for col in df.columns:
            # 🔹 Convert mixed-type categorical columns to strings
            if df[col].dtype == "object" or df[col].nunique() < 15:
                df[col] = df[col].astype(str)

                # Fill missing values with most frequent category (safe for mixed types)
                most_frequent = df[col].mode().astype(str)[0]
                df[col].fillna(most_frequent, inplace=True)
            else:
                # 🔹 Fill numerical missing values with median
                df[col].fillna(df[col].median(), inplace=True)

        # Encode categorical variables
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = df[col].astype(str)  # Ensure all categories are strings

            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                known_classes = set(self.encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else "unknown")

                # Update the encoder with the new "unknown" category
                self.encoders[col].classes_ = np.append(self.encoders[col].classes_, "unknown")
                df[col] = self.encoders[col].transform(df[col])

        # Feature-target split
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Save feature names (optional, if needed for reference)
        self.feature_names = X.columns.tolist()

        return X, y

    def train_initial_model(self, df, target_column, test_size=0.2, random_state=42):
        """Trains the model from scratch and returns performance metrics."""
        X, y = self.preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Store class labels
        if self.classes is None:
            self.classes = np.unique(y_train)

        # Train the model from scratch
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_prob = self.model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix
        }

    def generate_visualizations(self, df, target_column=None, y_pred=None, predictor_variables=None):
        """Generates useful graphs and returns them as Base64 strings for display in HTML."""
        graph_images = []

        if predictor_variables:
            df = df[predictor_variables + ([target_column] if target_column in df.columns else [])]

        # Identify numerical & categorical columns
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Ensure categorical columns are detected
        if not categorical_cols:
            categorical_cols = [col for col in df.columns if df[col].nunique() < 15 and df[col].dtype != "number"]

        # Function to convert plot to Base64
        def plot_to_base64():
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches="tight")
            plt.close()
            img.seek(0)
            return base64.b64encode(img.getvalue()).decode('utf8')

        # Correlation Heatmap
        if len(numerical_cols) > 1:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Heatmap")
            graph_images.append(plot_to_base64())

        # Churn Distribution (Actual vs Predicted)
        if target_column and target_column in df.columns:
            plt.figure(figsize=(12, 5))

            # Actual Distribution
            plt.subplot(1, 2, 1)
            sns.countplot(x=df[target_column], palette="coolwarm")
            plt.title("Actual Churn Distribution")
        
            # Predicted Distribution (if y_pred is provided)
            if y_pred is not None:
                plt.subplot(1, 2, 2)
                sns.countplot(x=y_pred, palette="coolwarm")
                plt.title("Predicted Churn Distribution")

            graph_images.append(plot_to_base64())

        # Histograms for Numerical Features
        for col in numerical_cols[:2]:  # Limit to first 2 numerical columns
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            graph_images.append(plot_to_base64())

        # Boxplots for Numerical Features
        for col in numerical_cols[:2]:  # Limit to first 2 numerical columns
            plt.figure(figsize=(6, 4))
            sns.boxplot(y=df[col])
            plt.title(f"Boxplot of {col}")
            graph_images.append(plot_to_base64())

        # Count Plots for Categorical Variables
        for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df[col], order=df[col].value_counts().index)
            plt.xticks(rotation=45)
            plt.title(f"Count of {col}")
            graph_images.append(plot_to_base64())

        # Churn vs Numerical Features (Actual vs Predicted)
        if target_column and target_column in df.columns and numerical_cols:
            for col in numerical_cols[:2]:
                plt.figure(figsize=(12, 5))

                # Actual Churn vs Numerical Feature
                plt.subplot(1, 2, 1)
                sns.boxplot(x=df[target_column], y=df[col])
                plt.title(f"Actual {col} vs {target_column}")

                # Predicted Churn vs Numerical Feature (if y_pred is provided)
                if y_pred is not None:
                    plt.subplot(1, 2, 2)
                    sns.boxplot(x=y_pred, y=df[col])
                    plt.title(f"Predicted {col} vs Churn")

                graph_images.append(plot_to_base64())

        # Confusion Matrix (if y_pred and target_column are provided)
        if target_column and y_pred is not None:
            plt.figure(figsize=(6, 4))
            cm = confusion_matrix(df[target_column], y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.classes, yticklabels=self.classes)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            graph_images.append(plot_to_base64())

        # ROC Curve (if y_pred and target_column are provided)
        if target_column and y_pred is not None:
            plt.figure(figsize=(6, 4))
            fpr, tpr, _ = roc_curve(df[target_column], y_pred)
            plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            graph_images.append(plot_to_base64())

        # Feature Count Plot
        if len(df.columns) > 10:
            plt.figure(figsize=(10, 5))
            df.nunique().sort_values(ascending=False)[:10].plot(kind="barh", color="skyblue")
            plt.title("Top 10 Features by Unique Values")
            graph_images.append(plot_to_base64())

        # Random Column Distribution
        if numerical_cols:
            col = np.random.choice(numerical_cols)
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Random Column Distribution: {col}")
            graph_images.append(plot_to_base64())

        # Pairplot for Small Datasets
        if len(numerical_cols) > 1 and df.shape[0] < 500:
            sns.pairplot(df[numerical_cols])
            graph_images.append(plot_to_base64())

        # Final Boxplot Comparison
        if categorical_cols and numerical_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[categorical_cols[0]], y=df[numerical_cols[0]])
            plt.title(f"{categorical_cols[0]} vs {numerical_cols[0]}")
            graph_images.append(plot_to_base64())

        return graph_images[:10]  # Return up to 10 graphs


class SalesPrediction:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.encoders = {}

    def prepare_time_series(self, df, target_column):
        """Prepares any dataset for time series analysis"""
        df = df.copy()
        
        # Ensure target is numeric
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        df.dropna(subset=[target_column], inplace=True)
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        if len(datetime_cols) > 0:
            # Use the first datetime column as index
            df = df.set_index(datetime_cols[0]).sort_index()
            # Ensure proper frequency
            try:
                df = df.asfreq(pd.infer_freq(df.index) or 'D')
            except:
                df = df.asfreq('D')
        else:
            # Create numeric index if no datetime column exists
            df.index = pd.RangeIndex(start=0, stop=len(df))
        
        return df[[target_column]].dropna()

    def train_initial_model(self, df, target_column, test_size=0.2, random_state=42):
        """Automated ARIMA modeling for any dataset"""
        try:
            # Prepare the time series data
            ts_data = self.prepare_time_series(df, target_column)
            
            # Check for sufficient data
            if len(ts_data) < 30:
                raise ValueError("Insufficient data points (minimum 30 required)")
            
            # Train-test split (preserve order)
            train_size = int(len(ts_data) * (1 - test_size))
            train, test = ts_data.iloc[:train_size], ts_data.iloc[train_size:]
            
            # Detect seasonality
            seasonal = False
            seasonal_period = 12  # Default monthly seasonality
            
            if len(train) > 50:
                try:
                    decomposition = seasonal_decompose(train, model='additive', period=seasonal_period)
                    seasonal = decomposition.seasonal.std() > 0.1 * decomposition.observed.std()
                except:
                    seasonal = False
            
            # Fit ARIMA model with error handling
            try:
                model = auto_arima(
                    train,
                    seasonal=seasonal,
                    m=seasonal_period if seasonal else None,
                    suppress_warnings=True,
                    stepwise=True,
                    trace=True,
                    error_action='ignore',
                    max_order=5,
                    maxiter=30,
                    n_jobs=1,
                    random_state=random_state
                )
                
                # Generate forecast
                forecast, conf_int = model.predict(
                    n_periods=len(test),
                    return_conf_int=True
                )
                
                # Create forecast index
                if isinstance(train.index, pd.DatetimeIndex):
                    last_date = train.index[-1]
                    freq = pd.infer_freq(train.index) or 'D'
                    forecast_index = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=len(test),
                        freq=freq
                    )
                else:
                    forecast_index = pd.RangeIndex(
                        start=train.index[-1] + 1,
                        stop=train.index[-1] + 1 + len(test)
                    )
                
                # Prepare results
                results = {
                    "model": model,
                    "metrics": {
                        "rmse": np.sqrt(mean_squared_error(test, forecast)),
                        "mae": mean_absolute_error(test, forecast),
                        "r2": r2_score(test, forecast),
                        "aic": model.aic() if hasattr(model, 'aic') else None,
                        "bic": model.bic() if hasattr(model, 'bic') else None,
                        "is_seasonal": seasonal,
                        "model_order": str(model.order),
                        "seasonal_order": str(model.seasonal_order) if hasattr(model, 'seasonal_order') else None
                    },
                    "forecast": forecast.tolist(),
                    "conf_int": conf_int.tolist() if conf_int is not None else None,
                    "summary": model.summary().as_text() if hasattr(model, 'summary') else "No summary available",
                    "forecast_index": forecast_index.astype(str).tolist()
                }
                
                return results
                
            except Exception as e:
                raise ValueError(f"ARIMA modeling failed: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"Data preparation failed: {str(e)}")

    def generate_visualizations(self, df, target_column, forecast=None, conf_int=None, model=None):
        """Generates visualization data that works with any dataset"""
        try:
            # Prepare the time series data
            ts_data = self.prepare_time_series(df, target_column)
            labels = ts_data.index.astype(str).tolist()
            
            # Initialize results structure
            results = {
                "actual_vs_predicted": {
                    "labels": labels,
                    "actual": ts_data[target_column].tolist(),
                    "predicted": [None] * len(ts_data)
                },
                "forecast": {
                    "labels": labels,
                    "historical": ts_data[target_column].tolist(),
                    "forecast": [None] * len(ts_data),
                    "upper_bound": [None] * len(ts_data),
                    "lower_bound": [None] * len(ts_data)
                }
            }
            
            # Add forecast data if available
            if forecast and isinstance(forecast, dict):
                # Combine historical and forecast data
                historical = ts_data[target_column].tolist()
                forecast_values = forecast.get('forecast', [])
                forecast_index = forecast.get('forecast_index', [])
                
                # Create full series for visualization
                full_series = historical + forecast_values
                full_labels = labels + forecast_index
                
                # Update results
                results["actual_vs_predicted"]["predicted"] = historical + forecast_values
                results["actual_vs_predicted"]["labels"] = full_labels
                
                results["forecast"]["forecast"] = [None] * len(historical) + forecast_values
                results["forecast"]["labels"] = full_labels
                
                # Add confidence intervals if available
                if forecast.get('conf_int'):
                    ci_lower = [x[0] for x in forecast['conf_int']]
                    ci_upper = [x[1] for x in forecast['conf_int']]
                    
                    results["forecast"]["upper_bound"] = [None] * len(historical) + ci_upper
                    results["forecast"]["lower_bound"] = [None] * len(historical) + ci_lower
            
            return results
            
        except Exception as e:
            print(f"Visualization generation failed: {str(e)}")
            return {
                "actual_vs_predicted": {"labels": [], "actual": [], "predicted": []},
                "forecast": {"labels": [], "historical": [], "forecast": [], "bounds": []}
            }



class Arima:
    def arma(self, df):
        pass

class RandomForest():
    def RandomForest(df):
        pass
