import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from azureml.core import Workspace, Dataset, Datastore, Run
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def preprocess_and_register_data():
    workspace = Workspace.from_config(path="config.json")
    # workspace = Workspace(
    #     subscription_id="c2a0e35d-38a9-4a75-bc6e-9e53de1fee8d",
    #     resource_group="tnt-resource-group",
    #     workspace_name="tnt-ml"
    # )
    
    # raw_data = Dataset.get_by_name(workspace=workspace,name='diabetes-data',version=2)
    # df = raw_data.to_pandas_dataframe()
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    data_asset = ml_client.data.get("diabetes-data", version="2")
    df = pd.read_csv(data_asset.path)

    with mlflow.start_run(run_name="Preprocessing Data"):
        # Log dataset parameters
        mlflow.log_param("data_columns", df.columns.tolist())
        mlflow.log_param("record_count", len(df))

        # Compute and log dataset metrics
        df_description = df.describe()
        mlflow.log_metrics({"mean_feature1": df_description.loc['mean', 'feature1']})

        # Data preprocessing steps
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df.drop('target', axis=1))
        df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
        df_scaled['target'] = df['target']

        # Save and log scaler
        scaler_path = 'outputs/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "model_artifacts")

        # Save a sample of processed data
        sample_data_path = 'outputs/sample_processed_data.csv'
        df_scaled.sample(5).to_csv(sample_data_path)
        mlflow.log_artifact(sample_data_path)

        # Create and log plots
        plt.figure(figsize=(10, 6))
        df['feature1'].hist(bins=30, alpha=0.5, color='blue', label='Original')
        df_scaled['feature1'].hist(bins=30, alpha=0.5, color='red', label='Scaled')
        plt.legend()
        plot_path = 'outputs/feature1_distribution.png'
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, "plots")

        # Register the processed data
        datastore = Datastore.get_default(workspace)
        processed_data_path = 'outputs/processed_data.csv'
        df_scaled.to_csv(processed_data_path, index=False)
        datastore.upload_files(files=[processed_data_path], target_path='processed-data/', overwrite=True)
        processed_data = Dataset.Tabular.from_delimited_files(path=[(datastore, 'processed-data/processed_data.csv')])
        processed_data = processed_data.register(workspace=workspace,
                                                 name='processed-data',
                                                 description='Processed data for model training',
                                                 create_new_version=True)

if __name__ == "__main__":
    preprocess_and_register_data()
