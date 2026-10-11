import pandas as pd
import joblib
from src.v1.monitor import monitor_model

ver = 1
date = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

def model_monitor(ver=1, date=date):
    # Load the reference and current data
    ref_data = pd.read_parquet(f'data/processed/v{ver}/x_train_processed_std.parquet')
    cur_data = pd.read_parquet(f'data/processed/v{ver}/cur_data_v{ver}.parquet')
    #^this doesn't exist yet (so using test data - look at api more for details),
    # but will be created in the future when we have new data to monitor.

    # Load the models
    dtc = joblib.load(f'models/models/v{ver}/DecisionTreeClassifier_best_{ver}.pkl')
    lr = joblib.load(f'models/models/v{ver}/LogisticRegression_best_{ver}.pkl')
    rfc = joblib.load(f'models/models/v{ver}/RandomForest_best_{ver}.pkl')
    xgbc = joblib.load(f'models/models/v{ver}/XGBClassifier_best_{ver}.pkl')

    # Monitor each model
    monitor_model(ref_data, cur_data, predictions=None, labels=None, version=1)
    
model_monitor(ver=ver, date=date)