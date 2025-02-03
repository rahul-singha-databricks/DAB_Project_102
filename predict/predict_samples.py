# Databricks notebook source
##################################################################################
# Set up widget to select the environment (dev, staging, prod)
##################################################################################

dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# COMMAND ----------
##################################################################################
# Import necessary libraries for MLflow deployment
##################################################################################

import mlflow.deployments
import os

# COMMAND ----------
##################################################################################
# Initialize MLflow deployment client
##################################################################################

client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------
##################################################################################
# Send prediction request to deployed MLflow model
##################################################################################

# Define input data for prediction
response = client.predict(
    endpoint=f"{env}_titanic_model-endpoint",  # Target endpoint for model inference
    inputs={
        "dataframe_split": {  # Use "dataframe_split" format for structured input
            "index": [0, 1],  # Row indices
            "columns": [  # Column names (features used in the model)
                "PassengerId",
                "Pclass",
                "Name",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked",
            ],
            "data": [  # Sample input data for prediction
                [
                    892,
                    3,
                    "Rahul Singha",
                    "male",
                    29.0,
                    0,
                    0,
                    232425,
                    31.3875,
                    "C23 C25 C27",
                    "Q",
                ],
                [
                    893,
                    3,
                    "Ayushi Kumar",
                    "female",
                    44.0,
                    0,
                    0,
                    232425,
                    31.3875,
                    "C23 C25 C27",
                    "Q",
                ],
            ],
        }
    },
)

# Print model prediction response
print(response)
