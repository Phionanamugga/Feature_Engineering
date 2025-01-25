import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd  # Add this import to resolve the NameError
from sklearn.model_selection import train_test_split



@pytest.fixture
def data():
    df = pd.read_csv("https://raw.githubusercontent.com/m-mahdavi/teaching/refs/heads/main/datasets/adult.csv")
    
    # Fill missing values with the median for numerical columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    # Update the target column to match the correct one after encoding
    target_column = "target_ >50K"
    
    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test


def test_categorical_and_numerical_columns(data):
    x_train, x_test, y_train, y_test = data
    # Your test code that works with x_train

def test_transformer_output_shape(data):
    x_train, x_test, y_train, y_test = data
    # Your test code that works with x_train and x_test

