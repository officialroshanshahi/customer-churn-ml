import pandas as pd
from ..v1.train import trainer
from scipy.stats import loguniform, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

#EXTRA CODE
ver = 1

def train_models(ver=1):
    #importing necessary processed data and exporting the trained model - EXTRA CODE
    x_train_processed = pd.read_parquet(f"data/processed/v{ver}/x_train_processed.parquet")
    x_train_processed_std = pd.read_parquet(f"data/processed/v{ver}/x_train_processed_std.parquet")
    y_train = pd.read_parquet(f"data/processed/v{ver}/y_train.parquet").values.ravel()

    model_dic = {
      'RandomForest': RandomForestClassifier(random_state=42),
      'LogisticRegression': LogisticRegression(max_iter=100, random_state=42),
      'XGBClassifier':XGBClassifier(random_state=42, eval_metric='logloss'),
      'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42)
    }

    #as logistic regression did not converge with saga or sag solver with max_iter=1000 or even 10000
    #best way to handle this would be use non-correlated feature dataset i.e. drop correlated features.
    #and data doesn't seem to be unbalanced either. but even if it was, it should have been handled by stratefy=y in train_test_split
    #and of course, class_weight could have been used too.
    #but since XGBClassifier is the best classifier among those listed above, i decided not to pursue it
    #XGBC will be used in production and not LR. so, I will just use sag solver for LR and not worry about it.
    model_lr = {
      'LogisticRegression': LogisticRegression(max_iter=5, l1_ratio=0, random_state=42)
    }

    params ={
      'RandomForest': {
        'n_estimators': list(range(50, 501, 50)),
        'max_depth': [None, 5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
      },
      'LogisticRegression': {   
        "solver": ["saga"],
        "C": loguniform(1e-4, 1e2),
        "l1_ratio": uniform(0.01, 0.98), #basically elastic net
        "tol": loguniform(1e-5, 1e-3),
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"]
      },
      'XGBClassifier': {
        'n_estimators': list(range(50, 501, 50)),
        'max_depth': list(range(3, 11)),
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
      },
      'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 6, 9, 12, 15, 18],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
      }
    }

    params_lr ={
      'LogisticRegression': {   
        "solver": ["sag"],
        "C": loguniform(1e-4, 1e2),
        "tol": loguniform(1e-5, 1e-3),
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"]
      }
    }

    scoring_metrics = {
      'accuracy': 'accuracy',
      'f1': 'f1',
      'roc_auc': 'roc_auc',
      'precision': 'precision',
      'recall': 'recall'
    }

    best_models = trainer(x_train_processed, x_train_processed_std, y_train, model_lr, params_lr, scoring_metrics, ver=ver)

train_models(ver=ver)