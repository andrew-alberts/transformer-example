import argparse
import pandas as pd

# Import the custom transformer classes we built.
from custom_impute_transformer import CustomImputeTransformer
from custom_impute_output_transformer import CustomImputeOutputTransformer

# Import miscellaneous other items from scikit-learn.
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Read the data file and define the feature and target columns.
    data = pd.read_csv("sample_rfm_data.csv")
    feature_names = [
    "NUM_DAYS_SINCE_LAST_TRANSACTION",
    "AMT_SPENT_LAST_TRANSACTION",
    "NUM_TRANSACTIONS_LAST_30_DAYS",
    "NUM_TRANSACTIONS",
    "NUM_ITEMS_PER_TRANSACTIONS",
    "AMT_SPENT_PER_TRANSACTION"
    ]
    target_name = "HAS_REWARDS_CARD"
    
    # Create the impute transformer object based on whether we want verbose output or not.
    impute_transformer = CustomImputeOutputTransformer()  if args.verbose else CustomImputeTransformer()
    
    # Create the Pipeline object which will sequentially fit the imputation and then fit the model.
    pipe = Pipeline([("Impute", impute_transformer), 
                     ("Model", LogisticRegression(solver="lbfgs"))])
    
    # Create the grid to search over. Grid search will try both "mean" and "median" imputation
    # and determine whether or not to fit an intercept to the logistic regression model.
    param_grid = {
        "Impute__impute_type": ["mean", "median"],
        "Model__fit_intercept": [True, False]
    }

    # Create a standard GridSearchCV object and use it to fit the best model.
    gs_cv = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, refit=True)
    gs_cv.fit(data[feature_names], data[target_name])

    # Get the best_estimator_ attribute which is the best pipeline and get the predictions on the trani set.
    best_pipe = gs_cv.best_estimator_
    print("Best pipeline", best_pipe)
    predictions = best_pipe.predict(data[feature_names])

    # Output the training accuracy of the pipeline we fit.
    training_accuracy = accuracy_score(y_true=data[target_name], y_pred=predictions)
    print("Training accuracy", training_accuracy)