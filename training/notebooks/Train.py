# Databricks notebook source
##################################################################################
%pip install --upgrade "mlflow-skinny[databricks]"
%pip install mlflow>=2.4.1
dbutils.library.restartPython()
# COMMAND ----------

# Set up Databricks widgets for selecting the environment and model name
##################################################################################

# Dropdown widget to select the environment (dev, staging, prod)
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# COMMAND ----------
# Text widget to input the model name (default: "titanic_model")
dbutils.widgets.text("model_name", defaultValue="titanic_model")
model_name = dbutils.widgets.get("model_name")

# Format model name with environment prefix
model_name = f"{env}_rs.project_102.{model_name}"
print(model_name)

# COMMAND ----------
##################################################################################
# Import necessary libraries for ML model training and evaluation
##################################################################################

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
##################################################################################
# Load dataset from Databricks table
##################################################################################

# Load the sample dataset into a Spark DataFrame
df = spark.table(f"{env}_rs.project_102.sample_data")

# COMMAND ----------
# Convert Spark DataFrame to Pandas DataFrame for model training
pdf = df.toPandas()

# Separate features (X) and target variable (y)
X = pdf.drop("Survived", axis=1)
y = pdf.Survived

# COMMAND ----------
##################################################################################
# Split dataset into training and test sets
##################################################################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# COMMAND ----------
# Display value counts for categorical feature "Pclass" in the training data
display(X_train["Pclass"].value_counts())

# COMMAND ----------
# Display value counts for categorical feature "Embarked" in the training data
display(X_train["Embarked"].value_counts())

# COMMAND ----------
##################################################################################
# Define preprocessing steps for categorical and numerical features
##################################################################################

# List of columns to drop from the dataset (not useful for model training)
dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin"]

# Define categorical and numerical feature columns
categorical_cols = ["Pclass", "Embarked", "Sex"]
numerical_cols = [
    col for col in X.columns if col not in dropped_cols + categorical_cols
]

# Pipeline for categorical features: Fill missing values and apply one-hot encoding
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Pipeline for numerical features: Fill missing values with mean and scale the data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the final pipeline with preprocessing and logistic regression model
model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

# COMMAND ----------
##################################################################################
# Train the logistic regression model
##################################################################################

model.fit(X_train, y_train)

# COMMAND ----------
##################################################################################
# Make predictions on the test set
##################################################################################

y_pred = model.predict(X_test)

# COMMAND ----------
# Print the predicted values
print(y_pred)

# COMMAND ----------
##################################################################################
# Evaluate model performance
##################################################################################

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred)

# Display confusion matrix and classification report
display(conf_matrix)
print(class_report)

# COMMAND ----------
##################################################################################
# Log model in MLflow and evaluate its performance
##################################################################################

# Prepare evaluation data
eval_data = X_test
eval_data["target"] = y_test

# Get an input example for MLflow model logging
input_example = X_train.iloc[[0]]

# Infer model signature
signature = infer_signature(X_train, model.predict(X_train))

# Start an MLflow run to log the model
with mlflow.start_run() as run:
    # Log the trained model to MLflow
    model_info = mlflow.sklearn.log_model(
        model,
        "logistic_regression_model",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature,
    )

    # Evaluate model using MLflow
    result = mlflow.evaluate(
        model_info.model_uri,
        eval_data,
        targets="target",
        model_type="classifier",
        evaluators=["default"],
    )
    