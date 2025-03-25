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
        self.encoders = {}

    def clean_sales_data(self, df, target_column=None):
        """Automatically cleans any sales dataset without knowing its structure."""    
        # Drop columns with excessive missing values (> 30%)
        df = df.dropna(thresh=int(0.7 * df.shape[1]))

        # Handle missing values dynamically
        for col in df.columns:
            if df[col].dtype == "object":
                df[col].fillna("Unknown", inplace=True)  # Fill categorical with "Unknown"
            else:
                df[col].fillna(df[col].median(), inplace=True)  # Fill numerical with median

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Detect and convert date columns
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

        # Extract features from date columns
        date_cols = df.select_dtypes(include=["datetime"]).columns
        for col in date_cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df.drop(columns=[col], inplace=True)

        # Standardize categorical text (lowercase, remove spaces)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.lower().str.strip()

        # Encode categorical variables
        label_encoders = {}
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoders for later use

        # Handle numerical outliers (IQR Method)
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
            df = df[~outliers.any(axis=1)]

        # Normalize numerical data
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        X = df.drop(columns=[target_column])
        y = df[target_column]


        return X, y
    
    def train_initial_model_v2(self, df):
        pass
    
    def generate_visualizations_v2(self, df):
        pass



# NA values
# outliers
# Missing Values


class Arima:
    def arma(self, df):
        pass

class RandomForest():
    def RandomForest(df):
        pass
