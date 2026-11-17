import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Ensure src is in the system path to load preprocess correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from preprocess import create_preprocessor

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def split_data(df, target_col="Failure Type"):
    """Separates features and target, and splits into train/test sets."""
    # Use 'Failure Type' or default to 'Machine failure' depending on dataset design
    if target_col not in df.columns:
        if 'Machine failure' in df.columns:
            target_col = 'Machine failure'
        else:
            # Dynamically infer target column as a fallback if expected columns are absent
            target_col = df.columns[-1]
            
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_model():
    """Creates and returns the machine learning Pipeline."""
    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        ))
    ])
    return model

def train_model(model, X_train, y_train):
    """Fits the pipeline model on training data."""
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Saves the trained pipeline using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model successfully saved to {filepath}")

def main():
    """Executes the complete end-to-end training pipeline."""
    # Default pathing based on typical project structures
    data_path = os.path.join("data", "predictive_maintenance.csv")
    model_path = os.path.join("models", "model.joblib")

    # Check for fallback data locations if standard is missing
    if not os.path.exists(data_path):
        # Check if there is any csv in data folder
        if os.path.exists("data"):
            csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
            if csv_files:
                data_path = os.path.join("data", csv_files[0])

    print(f"Loading data from: {data_path}")
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your dataset CSV file is placed in the 'data' directory.")
        return

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Building pipeline model...")
    model = build_model()

    print("Training model...")
    model = train_model(model, X_train, y_train)

    print("Saving trained model pipeline...")
    save_model(model, model_path)
    print("Workflow execution complete!")

if __name__ == '__main__':
    main()
