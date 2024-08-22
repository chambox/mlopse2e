import argparse
import glob
import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# define functions
def main(args):
    # MLflow: Start run
    mlflow.start_run()
    
    # Enable autologging
    mlflow.sklearn.autolog()

    # Read data
    df = read_csv_file(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    model = train_model(args.reg_rate, X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # End MLflow run
    mlflow.end_run()


def read_csv_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the file: {file_path}")
    if not file_path.endswith('.csv'):
        raise ValueError(f"The file is not a CSV: {file_path}")
    return pd.read_csv(file_path)



def split_data(df):
    X = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']].values
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, y_train):
    # Train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def parse_args():
    # Setup arg parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args

# Run script
if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Run main function
    main(args)
