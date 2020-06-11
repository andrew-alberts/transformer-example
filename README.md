# transformer-example

This is the code for the example presented in "Get the Most out of scikit-learn with Object-Oriented Programming."

To execute the pipeline for the simple impute transformer, run the following in the CLI: `python pipeline.py`

To execute the pipeline for the impute transformer with verbose output, run the following in the CLI: `python pipeline.py --verbose`

## File Descriptions

sample_rfm_data.csv: Sample customer-level recency, frequency, and monetary data.

pipeline.py: Script that drives the creation of the custom transformer and trains the model.

custom_impute_transformer.py: Defines the CustomImputeTransformer class. Objects of this class contain the fit and transform methods to create transformers to implement mean and median imputing. 

custom_impute_output_transformer.py: Defines the CustomImputeOutpuTransformer class. Objects of this class behave similar to objects of CustomImputeTransformer but print the summary statistics computed when fit is called.

verbose_output.txt: Verbose output of a call to pipeline.py using the --verbose flag to trigger building a pipeline with a CustomImputeOutpuTransformer transformer. 
