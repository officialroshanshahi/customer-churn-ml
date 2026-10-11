import os, glob, pandas as pd
from pathlib import Path
import numpy as np
from src.v1.preprocessing import preprocessor, load_data
from src.v1.monitor import monitor_model
from src.v1.main import retrain

# resolve the project root directory dynamically based on the current file's location
current_path = Path(__file__).resolve()
PROJECT_ROOT = next(p for p in current_path.parents if p.name == "customer-churn-ml")

# get all production data files for a given version, or fallback to test data if none are found
# it is only done for this project to make sure the model as well as retrain pipeline is working well
# in production the fallback can be removed
def get_all_prod_data(v):
    
    # finding the production data if any
    DIR_VER = PROJECT_ROOT / f"data/new/v{v}"
    files = glob.glob(os.path.join(DIR_VER, "production_data_*.csv"))
    
    # checking if there is production file if not fallback to test set
    if not files:
        
        print("\n[MONITORING INFO] No live production data files found in data/new/.")
        
        print("[MONITORING INFO] Falling back to Test Data Split as a temporary production proxy for testing...")
        
        # we only require test set so train set is _
        _, Xte, _, yt = load_data(ver=1)
        
        test_prod_proxy = Xte.copy()
        
        test_prod_proxy["churn"] = yt.values.ravel()
        
        return test_prod_proxy
    
    # returning the production data and making sure it doesn't throw error if data is corrupted or bad
    return pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in files], ignore_index=True)


# orchestrator for monitoring
def run_monitoring(model_instances):
    
    model, ver, name = model_instances
    prod = get_all_prod_data(ver)
    
    # static test baseline reference path lookup
    REF = PROJECT_ROOT / f"data/processed/v{ver}/test_baseline.csv"
    
    # AUTOMATIC FALLBACK: If test_baseline.csv is missing, generate it on the fly using the TEST split
    if not REF.exists():
        
        print(f"[MONITORING INFO] Baseline dataset missing at {REF}. Generating test_baseline.csv on the fly from test set...")
        
        _, Xte, _, yte = load_data(ver=ver)
        
        # preprocess test feature set to match baseline feature space
        meta_cols = {"churn", "request_id", "timestamp", "prediction", "probability", "risk_percentage", "risk_level", "threshold"}
        
        raw_te_cols = [c for c in Xte.columns if c not in meta_cols]
        
        x_te, _ = preprocessor(Xte[raw_te_cols], "no", "yes", ver)
        
        baseline = x_te.copy()
        
        #adding churn in the baseline as monitoring code requires x & y together
        baseline["churn"] = yte.squeeze()
        
        # saving the baseline for future use
        REF.parent.mkdir(parents=True, exist_ok=True)
        baseline.to_csv(REF, index=False)
        
        print(f"[MONITORING INFO] Saved test baseline dataset to {REF}")
        
        ref = baseline
    
    # if prod file exists
    else:
        ref = pd.read_csv(REF)
    
    # clean baseline ref target type to int
    if "churn" in ref.columns:
        ref["churn"] = pd.to_numeric(ref["churn"], errors="coerce").astype("int64")
    
    
    # some preprocessing logic to check if retrain is required
    prod.columns = [c.lower() for c in prod.columns]
    
    meta_cols = {"churn", "request_id", "timestamp", "prediction", "probability", "risk_percentage", "risk_level", "threshold"}
    raw_feature_cols = [c for c in prod.columns if c not in meta_cols]
    
    x_prod, xs_prod = preprocessor(prod[raw_feature_cols], "no", "yes", ver)
    eval_data = xs_prod if any(k in name.lower() for k in ["logistic", "kneighbor", "knc"]) else x_prod
    
    pred = model.predict_proba(eval_data)[:, 1]
    
    labels = None
    
    if "churn" in prod.columns:
        labels = pd.to_numeric(prod["churn"], errors="coerce").astype("int64")
    
    current_df = x_prod.copy()
    
    current_df["churn"] = labels
    
    result = monitor_model(ref, current_df, pred, labels, ver)
    
    # if retraining is required calling retrain function
    if result.get("retrain_required"): 
        result["retraining"] = retrain(ver)
    
    return result