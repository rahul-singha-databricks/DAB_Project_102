# Databricks notebook source
##################################################################################

dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")
dbutils.widgets.text("model_name", defaultValue="titanic_model")
model_name = dbutils.widgets.get("infa_table_name")
dbutils.widgets.text("infa_table_name", defaultValue="titanic_model")
infa_table_name = dbutils.widgets.get("infa_table_name")


# COMMAND ----------
import mlflow 
from mlflow.deployments import get_deploy_client

from mlflow.tracking import MlflowClient
client = MlflowClient(registry_uri="databricks-uc")
endpoint_client = get_deploy_client("databricks")
api_url = mlflow.utils.databricks_utils.get_webapp_url()
# COMMAND ----------

# Next we need an endpoint at which to execute our request which we can get from the Notebook's context
endpoint_name = f"{env}_{model_name}-endpoint"
model_name = f"{env}_rs.project_102.{model_name}"
print(endpoint_name)
print(model_name)

# COMMAND ----------
def get_champion_model_version(model_name):
    champion_version = 1
    for model in client.search_model_versions(f"name='{model_name}'"):
        mv = client.get_model_version(model_name, model.version)

        if mv.aliases == ['champion']:
            champion_version = model.version

    return champion_version


def get_endpoint_version(endpoint_name):
    try:
        endpoint = endpoint_client.get_endpoint(endpoint_name)
        print("Endpoint exists")
        return endpoint.config.served_models[0]['model_version']
    except:
        print("Endpoint does not exists")
        return -1

# COMMAND ----------
champion_version = get_champion_model_version(model_name)
endpoint_version = get_endpoint_version(endpoint_name)
print("champion_version = ",champion_version)
print("endpoint_version = ",endpoint_version)
# COMMAND ----------

if endpoint_version == -1:
    print("Endpoint does not exist")
    print("Endpoint is getting created")
    spark.sql(f"DROP TABLE IF EXISTS {env}_rs.project_102.{infa_table_name}_payload")
    endpoint = endpoint_client.create_endpoint(
        name= endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": f"{env}_rs.project_102.titanic_model",
                    "entity_version": champion_version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
            "traffic_config": {
                "routes": [
                    {"served_model_name": f"titanic_model-{champion_version}", "traffic_percentage": 100}
                ]
            },
            "auto_capture_config": {
                "catalog_name": f"{env}_rs",
                "schema_name": "project_102",
                "table_name_prefix": infa_table_name,
            },
        },
    )

elif endpoint_version == champion_version:
    print("No need to update the Endpoint")
else:
    print("Endpoint updated to new version")
    endpoint_client.update_endpoint(
        endpoint=endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": model_name,
                    "entity_version": champion_version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                }
            ],
            "traffic_config": {
                "routes": [
                    {"served_model_name": f"titanic_model-{champion_version}", "traffic_percentage": 100}
                ]
            },
            "auto_capture_config": {
                "catalog_name": f"{env}_rs",
                "schema_name": "project_102",
                "table_name_prefix": infa_table_name,
            },
        },
    )
#added test commit
