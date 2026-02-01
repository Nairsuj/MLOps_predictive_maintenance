# for data manipulation
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, confusion_matrix

# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import time

# =============================
# MLflow setup
# =============================

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

repo_id = "nairsuj/predictive-maintenance"

# Download file safely from HF Hub
Xtrain_path = hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_path = hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_path = hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Column Categorization
# =============================

numeric_features = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Scale positive weight: {class_weight:.3f}")

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# =============================
# Training + Evaluation
# =============================

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # =============================
    # Print ALL GridSearch results
    # =============================

    print("\n================ GRID SEARCH RESULTS ================")
    results_df = pd.DataFrame(grid_search.cv_results_)

    display_columns = [
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "params"
    ]

    print(results_df[display_columns].sort_values("rank_test_score"))

     # Log all parameter combinations and their mean test scores
    for i in range(len(results_df)):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results_df.loc[i, "params"])
            mlflow.log_metric("mean_cv_recall", results_df.loc[i, "mean_test_score"])
            mlflow.log_metric("std_cv_recall", results_df.loc[i, "std_test_score"])
        time.sleep(0.2)

   # =============================
    # Best model info
    # =============================

    print("\n================ BEST MODEL =================")
    print(f"Best CV Recall: {grid_search.best_score_:.4f}")
    print("Best Parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_recall", grid_search.best_score_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    # Predict on training set
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # Predict on test set
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    print("\n================ FINAL MODEL PERFORMANCE ================")

    print("\nTraining Set Classification Report:")
    print(classification_report(ytrain, y_pred_train))

    print("\nTest Set Classification Report:")
    print(classification_report(ytest, y_pred_test))

    print("\nTraining Confusion Matrix:")
    print(confusion_matrix(ytrain, y_pred_train))

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(ytest, y_pred_test))

   # Generate classification report
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })


    # Save the model locally
    model_path = "predictive_maintenance_model.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "nairsuj/predictive-maintenance"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="predictive_maintenance_model.joblib",
        path_in_repo="predictive_maintenance_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
