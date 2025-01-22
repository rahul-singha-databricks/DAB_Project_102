# Databricks notebook source
##################################################################################
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")
# COMMAND ----------
import mlflow.deployments
import os

# COMMAND ----------

client = mlflow.deployments.get_deploy_client("databricks")


response = client.predict(
    endpoint=f"{env}_titanic_model-endpoint",
    inputs={
        "dataframe_split": {
            "index": [0, 1],
            "columns": [
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
            "data": [
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
                    892,
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
print(response)
