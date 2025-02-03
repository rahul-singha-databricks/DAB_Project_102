# Databricks notebook source
##################################################################################


dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")
dbutils.widgets.text("model_name", defaultValue= "titanic_model")
model_name = dbutils.widgets.get("model_name")
# COMMAND ----------

import importlib
import mlflow
import os
import mlflow.pyfunc
import tempfile
import traceback
import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
api_url = mlflow.utils.databricks_utils.get_webapp_url()
token = dbutils.secrets.get(scope = "scope_rs", key = "DATABRICKS_TOKEN")
# COMMAND ----------

def get_run_id(model_name,model_version):
    mv = client.get_model_version(model_name, model_version)
    return mv.run_id

def get_accuracy_score(run_id):
    import requests
    metrics_url = f"{api_url}/api/2.0/mlflow/runs/get?run_id={run_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(metrics_url, headers=headers)
    metrics = response.json().get("run", {}).get("data", {}).get("metrics", {})

    accuracy = next(item for item in metrics if item["key"] == "accuracy_score")
    return float(accuracy["value"])


def get_latest_model_version(model_name):
    latest_version = 1
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def get_champion_model(model_name):
    champion_version = 0
    for model in client.search_model_versions(f"name='{model_name}'"):
        mv = client.get_model_version(model_name, model.version)

        if mv.aliases == ['champion']:
            champion_version = model.version

    return champion_version


def set_champion_model(model_name, old_champion_version, latest_version):
    if old_champion_version == 0:
        client.set_registered_model_alias(model_name, "champion", latest_version)
    else:
        champion_run_id = get_run_id(model_name,old_champion_version)
        champion_accuracy = float(get_accuracy_score(champion_run_id))
        latest_run_id = get_run_id(model_name,latest_version)
        latest_accuracy = float(get_accuracy_score(latest_run_id))
        if latest_accuracy > champion_accuracy:
            client.delete_registered_model_alias(model_name, "champion")
            client.set_registered_model_alias(model_name, "champion", latest_version)
            print(f"Champion model updated to version {latest_version}")
            print(f"Champion model version {latest_version} is the current champion model")
        else:
            print(f"Champion model version {old_champion_version} is still the current champion model")


def validate(model):
    try:
        validation_df = spark.table(f"{env}_rs.project_102.sample_data")
        pdf = validation_df.toPandas()
        X = pdf.drop("Survived",axis=1)
        model.predict(X)
        print("Validation Successful")
    except Exception as e:
        print("Validation Failed")
        raise e

# COMMAND ----------


model_name = f"{env}_rs.project_102.{model_name}"
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(model_name)
print(model_uri)
print(model_version)

# COMMAND ----------
model = mlflow.pyfunc.load_model(model_uri)
validate(model)

# COMMAND ----------

champion_version = get_champion_model(model_name)
print("champion_version = ",champion_version)
latest_version = get_latest_model_version(model_name)
print("latest_version = ",latest_version)

# COMMAND ----------
set_champion_model(model_name, champion_version, latest_version)
