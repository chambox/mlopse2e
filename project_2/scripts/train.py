from azureml.core import Workspace, Dataset, Run
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

def train_model():
    run = Run.get_context()
    workspace = run.experiment.workspace
    dataset = Dataset.get_by_name(workspace, name='processed-data')
    df = dataset.to_pandas_dataframe()

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/diabetes_model.pkl')

if __name__ == "__main__":
    train_model()
