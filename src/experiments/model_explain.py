import os
import joblib
import pandas as pd
from ..v1.explain import explain_model
from sklearn.pipeline import Pipeline

ver = 1

def explain_models(ver=1):
    #shap for each of the models
    dtc = joblib.load(f'models/models/v{ver}/DecisionTreeClassifier_best_{ver}.pkl')
    lr = joblib.load(f'models/models/v{ver}/LogisticRegression_best_{ver}.pkl')
    rfc = joblib.load(f'models/models/v{ver}/RandomForest_best_{ver}.pkl')
    xgbc = joblib.load(f'models/models/v{ver}/XGBClassifier_best_{ver}.pkl')

    x_test = pd.read_parquet(f'data/processed/v{ver}/x_test_processed.parquet')
    x_test_std = pd.read_parquet(f'data/processed/v{ver}/x_test_processed_std.parquet')
    
    os.makedirs(f'plots/shap/v{ver}', exist_ok=True)


    explain_model(
        x_test,
        dtc,
        'DecisionTree',
        'tree',
        ver=ver
    )

    explain_model(
        x_test_std,
        lr,
        'LogisticRegression',
        'kernel',
        ver=ver
    )

    explain_model(
        x_test,
        rfc,
        'RandomForest',
        'tree',
        ver=ver
    )

    explain_model(
        x_test,
        xgbc,
        'XGBoost',
        'tree',
        ver=ver
    )


explain_models(ver=ver)