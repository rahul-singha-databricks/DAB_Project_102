# Databricks notebook source
##################################################################################
# Set up widgets to define environment and table name
##################################################################################

# Dropdown widget to select the environment (dev, staging, prod)
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# COMMAND ----------
# Text widget to specify the table name (default: "titanic_model_payload")
dbutils.widgets.text("infa_table_name", defaultValue="titanic_model_payload")
infa_table_name = dbutils.widgets.get("infa_table_name")

# COMMAND ----------
##################################################################################
# Import necessary libraries for MLflow deployment
##################################################################################

import mlflow
from mlflow.deployments import get_deploy_client

# COMMAND ----------
##################################################################################
# Retrieve API token and Databricks workspace URL
##################################################################################

# Fetch authentication token for API access (not best practice for production)
# Recommended approach: Store token securely in a secret scope
token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()

# Create authorization header using the retrieved token
headers = {"Authorization": f"Bearer {token}"}

# Retrieve the Databricks workspace API URL
api_url = mlflow.utils.databricks_utils.get_webapp_url()
print(f"Databricks API URL: {api_url}")

# COMMAND ----------
##################################################################################
# Define the model name based on the selected environment
##################################################################################

model_name = f"{env}_rs.project_102.titanic_model"
print(f"Model Name: {model_name}")

# COMMAND ----------
##################################################################################
# Attempt to delete an existing deployment endpoint and drop the table
##################################################################################

try:
    # Set MLflow registry URI to Databricks Unity Catalog
    mlflow.set_registry_uri("databricks-uc")

    # Get MLflow deployment client for Databricks
    client = get_deploy_client("databricks")

    # Delete the existing deployment endpoint
    endpoint = client.delete_endpoint(endpoint=f"{env}_titanic_model-endpoint")

    # Drop the associated inference table if it exists
    spark.sql(f"DROP TABLE IF EXISTS {env}.project_102.titanic_model_payload")

except Exception as e:
    print("Endpoint does not exist or encountered an error:", str(e))
