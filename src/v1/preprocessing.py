import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# figuring out the project root directory dynamically, regardless of where the script is run from
# but since this file is in src/v1/, we can safely go up two levels (parents[2]) to reach the project root
# this can written in config, but this is a more robust way to ensure that the project root is always correctly identified
PROJECT_ROOT = Path(__file__).resolve().parents[2] if '__file__' in locals() else Path.cwd()

# define categorical feature categories for ordinal encoding
sub_cat = ['Basic', 'Standard', 'Premium']
con_len_cat = ['Monthly', 'Quarterly', 'Annual']

# creating a custom transformer for feature engineering
# of course, function could have been created to do this,
# but using a class allows us to integrate it into a sklearn pipeline seamlessly
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        
        x_copy = X.copy()
        
        # creating new features based on existing ones
        # other features were created but dropped due to low importance (while using shap importance)
        # or high correlation with other features
        x_copy["tenure_per_age"] = (x_copy["tenure"] / (x_copy["age"] + 1)).astype('float32')
        x_copy["usage_per_tenure"] = (x_copy["usage_frequency"] / (x_copy["tenure"] + 1)).astype('float32')
        x_copy["support_calls_per_tenure"] = (x_copy["support_calls"] / (x_copy["tenure"] + 1)).astype('float32')
        x_copy["support_usage_ratio"] = (x_copy["support_calls"] / (x_copy["usage_frequency"] + 1)).astype('float32')
        x_copy["spend_per_tenure"] = (x_copy["total_spend"] / (x_copy["tenure"] + 1)).astype('float32')
        x_copy["churn_risk_signals"] = ((x_copy["last_interaction"] >= 30).astype(int) + (x_copy["usage_frequency"] <= 5).astype(int) + (x_copy["support_calls"] >= 5).astype(int) + (x_copy["payment_delay"] > 0).astype(int)).astype('int16')
        x_copy["customer_friction"] = (x_copy["support_calls"] + x_copy["payment_delay"]).astype('int16')
        
        # dropping the original 'payment_delay' column as it has been incorporated into new features
        x_copy.drop(columns='payment_delay', inplace=True, errors='ignore')
        
        return x_copy
    
    def get_feature_names_out(self, input_features=None):
        
        if input_features is None: return None
        
        # we want to ensure that the new features are appended to the original feature list, excluding 'payment_delay'
        features = [f for f in input_features if f != 'payment_delay']
        
        new_features = ["tenure_per_age",
                        "usage_per_tenure",
                        "support_calls_per_tenure",
                        "support_usage_ratio",
                        "spend_per_tenure",
                        "churn_risk_signals",
                        "customer_friction"]
        
        # we return the combined list of original and new features, ensuring that the new features are added at the end
        return np.array(features + new_features)

# lowering the size of the dataset by converting data types to more memory-efficient ones
# i tried many different data types and combinations (int8, int 16...), and these were the best for this dataset
def convert_data_type(x):
    
    x_copy = x.copy()
    
    # getting all columns except 'total_spend' to convert them to int32 (if possible)
    other_columns = x_copy.columns.drop('total_spend', errors='ignore')
    
    for column in other_columns:
        try:
            if x_copy[column].dtype in ['object', 'string']: continue
            x_copy[column] = x_copy[column].astype('int32')
        except Exception: pass
    
    #this can be int16 (if we look at the dataset ifself max value is within the range of int16,
    # but to be safe we can use float32 for total_spend)
    if 'total_spend' in x_copy.columns: x_copy['total_spend'] = x_copy['total_spend'].astype('float32')
    
    return x_copy

# renaming col to make things easier for us
def col_rename(x):
    x_copy = x.copy()
    
    x_copy.columns = [col.lower().replace(' ', '_') for col in x_copy.columns]
    
    x_copy.drop(columns='customerid', errors='ignore', inplace=True)
    
    return x_copy

# the main logic behind the preprocessing pipeline,
# which includes imputation, encoding, feature engineering, and scaling
# to avoid data leakage, we fit the pipeline on the training data and then transform the test data using the fitted pipeline
# use fit_transform='yes' for training data and fit_transform='no' for test data
def preprocessor(data, fit_transform='yes', std='yes', ver=1):
    
    data_copy = col_rename(data)
    
    # we will save the pipeline and scaler to disk for later use,
    # so we can apply the same transformations to new data during inference
    pipeline_path = PROJECT_ROOT / f'models/pipelines/v{ver}/pipeline_normal_v{ver}.joblib'
    scaler_path = PROJECT_ROOT / f'models/scalers/v{ver}/scaler_v{ver}.joblib'

    # if fit_transform is 'yes', we fit the pipeline and scaler on the training data and save them to disk
    if fit_transform.lower().strip() == 'yes':
        
        # if we look at the dataset there is only one isnull().sum()
        # which implies that they could have been dropped
        # but to be safe we will impute missing values for both categorical and numerical features
        impute_full = ColumnTransformer(transformers=[
            ('cat_', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_exclude='number', dtype_include=['string','object'])),
            ('num_', SimpleImputer(strategy='median'), make_column_selector(dtype_exclude='string', dtype_include='number'))
        ], verbose_feature_names_out=False, remainder='passthrough')
        
        # we will use ordinal encoding for 'subscription_type' and 'contract_length' as they have a natural order,
        # and one-hot encoding for 'gender' as it is nominal
        encode = ColumnTransformer(transformers=[
            ('enc_', OrdinalEncoder(categories=[sub_cat, con_len_cat]), ['subscription_type', 'contract_length']),
            ('enc', OneHotEncoder(drop='first', sparse_output=False), ['gender'])
        ], verbose_feature_names_out=False, remainder='passthrough')

        
        pipeline_normal = Pipeline(steps=[
            ('imputer', impute_full),
            ('encoder', encode),
            ('reduce_size', FunctionTransformer(convert_data_type)),
            ('feature_eng', CustomFeatureEngineer())])
        
        # we set the output of the pipeline to be a pandas DataFrame,
        # so we can easily access the feature names and work with the data
        pipeline_normal.set_output(transform="pandas")
        
        # we fit the pipeline on the training data and transform it, which will apply all the steps defined in the pipeline
        data_transformed = pipeline_normal.fit_transform(data_copy)
        
        # we save the fitted pipeline to disk so we can use it later to transform new data during inference (or for testing)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_normal, pipeline_path)

        # checking if we need to scale the data,
        # if yes we will fit the scaler on the training data and returning both the transformed and scaled data
        if std.lower().strip() == 'yes':
            
            # not scaling the categorical features as they are already encoded and scaling them would not make sense
            exclude_cols = ["subscription_type",
                            "contract_length",
                            "gender_Male",
                            "churn_risk_signals",
                            "customer_friction"]
            
            # we will scale only the numerical features,
            # and we will use StandardScaler to standardize the features by removing the mean and scaling to unit variance
            features_to_scale = [col for col in data_transformed.columns if col not in exclude_cols]
            
            scaler = ColumnTransformer(transformers=[
                ('scaler', StandardScaler(), features_to_scale)],
                remainder='passthrough',
                verbose_feature_names_out=False)
            
            
            scaler.set_output(transform="pandas")
            
            data_transformed_std = scaler.fit_transform(data_transformed)[data_transformed.columns]
            
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            
            return data_transformed, data_transformed_std
        
        return data_transformed
    
    # if fit_transform is 'no', we load the saved pipeline and scaler from disk and use them to transform the test data
    else:
        # ensuring that the pipeline exist before trying to load it, to avoid runtime errors
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}.")
        
        pipeline_normal = joblib.load(pipeline_path)
        pipeline_normal.set_output(transform="pandas")
        
        data_transformed = pipeline_normal.transform(data_copy)
        
        if std.lower().strip() == 'yes':
            # ensuring that the scaler file exists before trying to load it, to avoid runtime errors
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")
            
            scaler = joblib.load(scaler_path)
            
            scaler.set_output(transform="pandas")
            
            data_transformed_std = scaler.transform(data_transformed)[data_transformed.columns]
            
            return data_transformed, data_transformed_std
        
        return data_transformed


# preprocessing pipeline for the target variable (y), which includes imputation and type conversion
# same as preprocessor function, we fit the pipeline on the training data and then transform the test data using the fitted pipeline
# use fit_transform='yes' for training data and fit_transform='no' for test data
def y_preprocessor(y, fit_transform='yes', ver=1):
    
    y_copy = pd.DataFrame(y).copy()
    
    y_copy.columns = y_copy.columns.str.lower()
    
    # for saving the imputer to disk for later use, so we can apply the same transformations to new data during inference (or test)
    imputer_path = PROJECT_ROOT / f'models/imputers/v{ver}/y_imputer_v{ver}.joblib'
    
    if fit_transform.lower().strip() == 'yes':
        
        y_imputer = ColumnTransformer(transformers=[
            ('impute', SimpleImputer(strategy='most_frequent'), ['churn'])
        ], verbose_feature_names_out=False)
        
        y_imputer.set_output(transform="pandas")
        
        # lowering the size of the target variable by converting it to int8, as it is a binary variable (0 or 1)
        y_transformed = y_imputer.fit_transform(y_copy).astype('int8')
        
        # we save the fitted imputer to disk so we can use it later to transform new data during inference (or for testing)
        imputer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(y_imputer, imputer_path)
        
        return y_transformed
    else:
        # ensuring that the imputer file exists before trying to load it, to avoid runtime errors
        if not imputer_path.exists():
            raise FileNotFoundError(f"Imputer file not found at {imputer_path}.")
        
        y_imputer = joblib.load(imputer_path)
        
        y_imputer.set_output(transform="pandas")
        
        return y_imputer.transform(y_copy).astype('int8')


# loading the raw data, splitting it into training and test sets, and returning them
# if base_only is 'yes', we return only the training set with the target variable (y) included,
# otherwise we return both training and test sets separately
# it is for monitoring purposes, where we want to have a baseline dataset to compare against the current data
def load_data(ver=1, base_only='no'):
    
    data_1 = pd.read_csv(PROJECT_ROOT / "data/raw/customer_churn_dataset-testing-master.csv")
    data_2 = pd.read_csv(PROJECT_ROOT / "data/raw/customer_churn_dataset-training-master.csv")
    
    data_full = pd.concat([data_1, data_2])
    
    X, y = data_full.drop(columns='Churn'), data_full[['Churn']]
    
    y = y_preprocessor(y, fit_transform='yes', ver=ver)
    
    # stratify is being used here but since the dataset is already balanced (roughly 55/45), it won't have much effect on the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    # if base_only is 'yes', we return only the training set with the target variable (y) included,
    # otherwise we return both training and test sets separately
    return pd.concat([X_train, y_train], axis=1) if base_only == 'yes' else (X_train, X_test, y_train, y_test)
