import mlflow
import numpy as np

# Start an MLflow run
mlflow.start_run()

# Log a metric
mlflow.log_metric("test_metric", 1.23)

# Log a parameter
mlflow.log_param("test_param", "parameter_value")

# Log an artifact (a text file in this case)
with open("test.txt", "w") as f:
    f.write("Hello, MLflow!")
mlflow.log_artifact("test.txt")

# Log a dictionary of metrics
metrics = {"accuracy": 0.95, "precision": 0.9, "recall": 0.92}
mlflow.log_metrics(metrics)

# Log a numpy array as an artifact
np.save("array.npy", np.array([1, 2, 3]))
mlflow.log_artifact("array.npy")

# Log a model (here using a simple dictionary as a placeholder)
model = {'coef': np.random.randn(10), 'intercept': np.random.rand()}
np.save("model.npy", model)  # Normally you would use joblib or pickle to save a scikit-learn model
mlflow.log_artifact("model.npy")

# End the run
mlflow.end_run()
