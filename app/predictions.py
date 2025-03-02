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
        self.model = SGDClassifier(loss="log_loss", random_state=42)  # Supports incremental learning
        self.scaler = StandardScaler()
        self.encoders = {}  
        self.classes = None
        self.MODEL_PATH = "customer_churn/churn_model.pkl"
        self.ENCODERS_PATH = "customer_churn/encoders.pkl"

        # Ensure model directory exists
        if not os.path.exists("customer_churn"):
            os.makedirs("customer_churn")

        # Load previous model if exists
        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, "rb") as f:
                saved_model = pickle.load(f)
                self.model = saved_model.model
                self.scaler = saved_model.scaler
                self.encoders = saved_model.encoders
                self.classes = saved_model.classes

    def preprocess_data(self, df, target_column):
        """Handles missing values and encodes categorical features dynamically."""
        for col in df.columns:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        # Encode categorical variables dynamically
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])

        # Feature-target split
        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    def train_initial_model(self, df, target_column, test_size=0.2, random_state=42):
        """Trains the model incrementally and returns performance metrics."""
        X, y = self.preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Store class labels
        if self.classes is None:
            self.classes = np.unique(y_train)

        # Train the model incrementally
        self.model.partial_fit(X_train, y_train, classes=self.classes)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_prob = self.model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Save model and encoders
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self, f)
        with open(self.ENCODERS_PATH, "wb") as f:
            pickle.dump(self.encoders, f)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix
        }

    def generate_visualizations(self, df, target_column=None):
        """Generates exactly 10 useful graphs and returns them as Base64 strings for display in HTML."""
        graph_images = []

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

        # Churn Distribution
        if target_column and target_column in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df[target_column], palette="coolwarm")
            plt.title("Churn Distribution")
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

        # Churn vs Numerical Features
        if target_column and target_column in df.columns and numerical_cols:
            for col in numerical_cols[:2]:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=df[target_column], y=df[col])
                plt.title(f"{col} vs {target_column}")
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

        return graph_images[:10]


def LinearRegression(df):
    pass

def Arima(df):
    pass

def RandomForest(df):
    pass

def SVM(df):
    pass

def Decisiontree(df):
    pass