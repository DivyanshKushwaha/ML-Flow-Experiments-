from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn
import os



# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set experiment name
mlflow.set_experiment('breast-cancer-rf-hp')

# Start MLflow run
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # Log child runs for each hyperparameter combination
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("accuracy", best_score)

    # Save train and test data as CSV files
    os.makedirs("datasets", exist_ok=True)
    train_csv_path = "datasets/train_data.csv"
    test_csv_path = "datasets/test_data.csv"
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv(train_csv_path, index=False)

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_csv_path = "datasets/test_data.csv"
    test_df.to_csv(test_csv_path, index=False)

    # Log train and test datasets as artifacts
    mlflow.log_artifact(train_csv_path, artifact_path="datasets")
    mlflow.log_artifact(test_csv_path, artifact_path="datasets")

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Divyansh Kushwaha")

    print(best_params)
    print(best_score)









### Without hypertuning

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# import os

# # Load the Breast Cancer dataset
# data = load_breast_cancer()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target, name='target')

# # Splitting into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Creating the RandomForestClassifier model
# rf = RandomForestClassifier(random_state=42)

# # Defining the parameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [10, 50, 100],
#     'max_depth': [None, 20, 30]
# }

# # Applying GridSearchCV
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Set experiment name
# mlflow.set_experiment('breast-cancer-rf-hp')

# # Start MLflow run
# with mlflow.start_run():
#     # Fit the model
#     grid_search.fit(X_train, y_train)

#     # Get the best parameters and score
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_

#     # Log params
#     mlflow.log_params(best_params)

#     # Log metrics
#     mlflow.log_metric("accuracy", best_score)

#     # Save train and test data as CSV files
#     os.makedirs("datasets", exist_ok=True)
#     train_csv_path = "datasets/train_data.csv"
#     test_csv_path = "datasets/test_data.csv"

#     train_df = X_train.copy()
#     train_df['target'] = y_train
#     train_df.to_csv(train_csv_path, index=False)

#     test_df = X_test.copy()
#     test_df['target'] = y_test
#     test_df.to_csv(test_csv_path, index=False)

#     # Log train and test datasets as artifacts
#     mlflow.log_artifact(train_csv_path, artifact_path="datasets")
#     mlflow.log_artifact(test_csv_path, artifact_path="datasets")

#     # Log the best model
#     mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

#     # Set tags
#     mlflow.set_tag("author", "Divyansh Kushwaha")

#     # Print results
#     print("Best Parameters:", best_params)
#     print("Best Accuracy:", best_score)
