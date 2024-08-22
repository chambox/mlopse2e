#!/bin/bash

# Create directory structure
echo "Creating directory structure..."
mkdir -p project_2/data
mkdir -p project_2/scripts
mkdir -p project_2/models
mkdir -p project_2/environments
mkdir -p project_2/azureml

# Create Python scripts
echo "Creating Python scripts..."
cat <<EOF > project_2/scripts/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow

def preprocess_data(filepath):
    mlflow.start_run(run_name="Data Preprocessing")
    data = pd.read_csv(filepath)
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(data.drop('target', axis=1))
    target = data['target']

    # Log preprocessing details
    mlflow.log_params({"data_columns": data.columns.tolist(), "target_column": 'target'})
    mlflow.sklearn.log_model(scaler, "scaler")
    mlflow.end_run()
    
    return processed_data, target
EOF

cat <<EOF > project_2/scripts/train.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_and_evaluate():
    mlflow.start_run(run_name="Model Training")
    data = pd.read_csv('data/processed_data.csv')
    target = pd.read_csv('data/target.csv')
    
    clf = RandomForestClassifier()
    clf.fit(data, target)
    predictions = clf.predict(data)
    accuracy = accuracy_score(target, predictions)

    # Log model and accuracy
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")
    mlflow.end_run()
    
    joblib.dump(clf, 'models/model.pkl')

if __name__ == "__main__":
    train_and_evaluate()
EOF

cat <<EOF > project_2/scripts/score.py
import json
import joblib
import numpy as np

def init():
    global model
    model_path = "models/model.pkl"
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    prediction = model.predict(np.array(data['data']))
    return prediction.tolist()
EOF

# Create requirements.txt
echo "Creating requirements.txt..."
cat <<EOF > project_2/requirements.txt
pandas
scikit-learn
numpy
mlflow
azureml-sdk
joblib
EOF

# Create Azure ML YAML configurations
echo "Creating Azure ML configuration files..."
cat <<EOF > project_2/azureml/endpoint.yml
\$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
name: my-model-endpoint
auth_mode: key
deployments:
  - name: blue
    model: azureml:model@latest
    code_configuration:
      code: scripts/
      scoring_script: score.py
    environment: azureml:myenv:1
    instance_type: Standard_DS2_v2
    instance_count: 1
EOF

cat <<EOF > project_2/azureml/training_job.yml
\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: scripts/
command: python train.py
environment: azureml:myenv:1
compute: azureml:cpu-cluster
experiment_name: model-training
description: "Training job for RandomForest model"
EOF

cat <<EOF > project_2/azureml/dataset_registration.yml
name: raw_data
version: 1
description: Raw data for model training
path: azureml://datastores/blob_datastore/paths/data/raw_data.csv
type: uri_file
EOF

echo "Setup complete."

