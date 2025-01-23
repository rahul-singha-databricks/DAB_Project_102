# Databricks notebook source
##################################################################################

dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")
# COMMAND ----------
dbutils.widgets.text("infa_table_name", defaultValue="titanic_model_payload")
infa_table_name = dbutils.widgets.get("infa_table_name")

# COMMAND ----------
import mlflow
from mlflow.deployments import get_deploy_client

# COMMAND ----------

# We need both a token for the API, which we can get from the notebook.
# Recall that we discuss the method below to retrieve tokens is not the best practice. We recommend you create your personal access token and save it in a secret scope.
token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an endpoint at which to execute our request which we can get from the Notebook's context
api_url = mlflow.utils.databricks_utils.get_webapp_url()
print(api_url)

# COMMAND ----------
model_name = f"{env}_rs.project_102.titanic_model"
print(model_name)

# COMMAND ----------

try:
    mlflow.set_registry_uri("databricks-uc")
    client = get_deploy_client("databricks")

    endpoint = client.delete_endpoint(endpoint=f"{env}_titanic_model-endpoint")
    spark.sql(f"DROP TABLE IF EXISTS {env}.project_102.titanic_model_payload")
except:
    print("Endpoint does not exist")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

endpoint = client.create_endpoint(
    name=f"{env}_titanic_model-endpoint",
    config={
        "served_entities": [
            {
                "entity_name": f"{env}_rs.project_102.titanic_model",
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": "titanic_model-1", "traffic_percentage": 100}
            ]
        },
        "auto_capture_config": {
            "catalog_name": f"{env}_rs",
            "schema_name": "project_102",
            "table_name_prefix": infa_table_name,
        },
    },
)
# COMMAND ----------
