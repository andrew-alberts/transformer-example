import pandas as pd
from custom_impute_transformer import CustomImputeTransformer



class CustomImputeOutputTransformer(CustomImputeTransformer):
    def __init__(self, impute_type=None):
        CustomImputeTransformer.__init__(self, impute_type)
    
    
    def fit(self, X, y=None):
        # Compute the result of fit from the parent class.
        result = super(CustomImputeOutputTransformer, self).fit(X, y)
        
        # Print the imputation type and the imputation values computation during this call of fit.
        print("Impute type", self.impute_type)
        print("Impute vals", self.impute_vals)
        
        return result