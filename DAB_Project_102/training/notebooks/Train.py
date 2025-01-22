# Databricks notebook source
##################################################################################


dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# COMMAND ----------
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# COMMAND ----------
df = spark.table(f"{env}_rs.project_102.sample_data")

# COMMAND ----------
pdf = df.toPandas()
X = pdf.drop("Survived",axis=1)
y = pdf.Survived

# COMMAND ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------
display(X_train['Pclass'].value_counts())

# COMMAND ----------
display(X_train['Embarked'].value_counts())

# COMMAND ----------

dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
categorical_cols = ["Pclass", "Embarked", "Sex"]
numerical_cols = [col for col in X.columns if col not in dropped_cols + categorical_cols]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# COMMAND ----------

model.fit(X_train, y_train)


# COMMAND ----------
y_pred = model.predict(X_test)

# COMMAND ----------
print(y_pred)

# COMMAND ----------

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate classification report
class_report = classification_report(y_test, y_pred)

# Display results
display(conf_matrix)
print(class_report)


# COMMAND ----------
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log model and metrics with MLflow
mlflow.set_experiment(f"/Users/rahul.singha@databricks.com/{env}_rs.project_102.titanic_model")

with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    signature = infer_signature(X_train, y_train)
    input_example = X_train.iloc[[0]]
    mlflow.sklearn.log_model(model, "logistic_regression_model", registered_model_name = f"{env}_rs.project_102.titanic_model",input_example = input_example )
    # mlflow.register_model(f"runs:/{run_id}/model", "dev_rs.project_102.titanic_model")
