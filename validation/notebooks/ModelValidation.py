# Databricks notebook source
##################################################################################
%pip install --upgrade "mlflow-skinny[databricks]"
%pip install mlflow>=2.4.1
dbutils.library.restartPython()
# COMMAND ----------
# Initialize widgets for selecting environment and model name
##################################################################################

# Dropdown widget for selecting the environment (dev, staging, prod)
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Text widget for specifying the model name (default: "titanic_model")
dbutils.widgets.text("model_name", defaultValue="titanic_model")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

##################################################################################
# Import necessary libraries and initialize MLflow client
##################################################################################

import importlib
import mlflow
import os
import mlflow.pyfunc
import tempfile
import traceback
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")

# Initialize MLflow Client
client = MlflowClient()

# Fetch Databricks API URL
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# Fetch authentication token from Databricks secrets
token = dbutils.secrets.get(scope="scope_rs", key="DATABRICKS_TOKEN")

# COMMAND ----------

##################################################################################
# Define helper functions for retrieving model details and accuracy
##################################################################################


def get_run_id(model_name, model_version):
    """
    Retrieves the run ID for a specific model version.
    """
    mv = client.get_model_version(model_name, model_version)
    return mv.run_id


def get_accuracy_score(run_id):
    """
    Fetches the accuracy score of a model using its run ID.
    """
    import requests

    metrics_url = f"{api_url}/api/2.0/mlflow/runs/get?run_id={run_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(metrics_url, headers=headers)
    metrics = response.json().get("run", {}).get("data", {}).get("metrics", {})

    # Extract accuracy score from metrics
    accuracy = next(item for item in metrics if item["key"] == "accuracy_score")
    return float(accuracy["value"])


def get_latest_model_version(model_name):
    """
    Retrieves the latest version of the given model.
    """
    latest_version = 1
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_champion_model(model_name):
    """
    Retrieves the version number of the current champion model.
    """
    champion_version = 0
    for model in client.search_model_versions(f"name='{model_name}'"):
        mv = client.get_model_version(model_name, model.version)

        # Check if the model is labeled as "champion"
        if mv.aliases == ["champion"]:
            champion_version = model.version

    return champion_version


def set_champion_model(model_name, old_champion_version, latest_version):
    """
    Updates the champion model if the latest version has a higher accuracy score.
    """
    if old_champion_version == 0:
        # If no champion exists, set the latest version as champion
        client.set_registered_model_alias(model_name, "champion", latest_version)
    else:
        # Compare accuracy scores of current champion and latest model
        champion_run_id = get_run_id(model_name, old_champion_version)
        champion_accuracy = float(get_accuracy_score(champion_run_id))
        latest_run_id = get_run_id(model_name, latest_version)
        latest_accuracy = float(get_accuracy_score(latest_run_id))

        if latest_accuracy > champion_accuracy:
            # Update champion model to the latest version
            client.delete_registered_model_alias(model_name, "champion")
            client.set_registered_model_alias(model_name, "champion", latest_version)
            print(f"Champion model updated to version {latest_version}")
        else:
            print(
                f"Champion model version {old_champion_version} is still the current champion model"
            )


def validate(model):
    """
    Validates the given model using a test dataset from Databricks tables.
    """
    try:
        # Load validation dataset from Databricks table
        validation_df = spark.table(f"{env}_rs.project_102.sample_data")
        pdf = validation_df.toPandas()
        X = pdf.drop("Survived", axis=1)

        # Run predictions using the model
        model.predict(X)
        print("Validation Successful")
    except Exception as e:
        print("Validation Failed")
        raise e


# COMMAND ----------

##################################################################################
# Load the latest model version and print details
##################################################################################

# Construct model name and retrieve latest version
model_name = f"{env}_rs.project_102.{model_name}"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"

# Print model details
print(f"Model Name: {model_name}")
print(f"Model URI: {model_uri}")
print(f"Model Version: {model_version}")

# COMMAND ----------

##################################################################################
# Load and validate the model
##################################################################################

# Load model from MLflow registry
model = mlflow.pyfunc.load_model(model_uri)

# Validate the model on test data
validate(model)

# COMMAND ----------

##################################################################################
# Retrieve and print champion and latest model versions
##################################################################################

champion_version = get_champion_model(model_name)
print(f"Champion Model Version: {champion_version}")

latest_version = get_latest_model_version(model_name)
print(f"Latest Model Version: {latest_version}")

# COMMAND ----------

##################################################################################
# Update the champion model if needed
##################################################################################

set_champion_model(model_name, champion_version, latest_version)
print("Champion model selection process completed.")
#test comment
