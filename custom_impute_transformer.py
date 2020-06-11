import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin



class CustomImputeTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, impute_type=None):
        # Set initial values for the attributes of the class.
        # We will expect impute_type to be one of "mean" or "median"
        self.impute_vals = None
        self.impute_type = impute_type
        
    
    def fit(self, X, y=None):
        # Compute the values for imputation using the method select at time of class init.
        if self.impute_type == "median":
            self.impute_vals = X.median()
        else:
            self.impute_vals = X.mean()
            
        return self


    def transform(self, X):
        # Fill missing values using the summary statistics computed when fit was run.
        return X.fillna(self.impute_vals)
    

    
if __name__ == "__main__":
    # Create a CustomImputeTransformer object which will perform mean imputation.
    cit = CustomImputeTransformer(impute_type="mean")
    
    # Fit the transformer on sample data and display the means computed.
    X_test = pd.DataFrame({"x1":[1, 2, np.nan, 4], "x2": [np.nan, np.nan, -1, 1]})
    cit.fit(X_test)
    print(cit.impute_vals)
    
    # Display the result of applying the mean transformer to the data.
    print(cit.transform(X_test))