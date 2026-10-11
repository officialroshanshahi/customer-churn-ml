import json
import pandas as pd
import uuid
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.v1.preprocessing import load_data, preprocessor
from src.v1.train import trainer
from src.v1.evaluate import evaluator
from src.v1.config import (
    TARGET, MIN_AUC, BASE, ACTIVE, MODELS, 
    PARAMS, SCORES, PROD, ver, MIN_LABELS, DECISION_THRESHOLD
)

# project/src/v1/main.py -> parents[2] is project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# this function resolves relative paths against the project root,
# ensuring that all file paths used in the application are absolute
# and correctly point to the intended files or directories.
def get_resolved_path(path_str: str) -> Path:
    """safely resolves relative config paths against the project root."""
    
    path_obj = Path(path_str)
    
    # if the path is already absolute, return it as is; otherwise, resolve it against the project root
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj).resolve()


# this function handles the entire training pipeline,
# including data loading, preprocessing, model training, evaluation, and saving the best model.
# if retrain is set to 'yes', it will retrain the model using the provided data;
# otherwise, it will use the default training data.
def train_pipeline(ver=1, retrain='no', data=None):
    
    if retrain == 'yes' and data is not None:
        X = data.drop(columns=[TARGET])
        y = data[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    else:
        Xtr, Xte, ytr, yte = load_data(ver=ver, base_only='no')

    # preprocessing
    X_train_processed, X_train_std = preprocessor(Xtr, "yes", "yes", ver)
    X_test_processed, X_test_std = preprocessor(Xte, "no", "yes", ver)

    # training
    models = trainer(X_train_processed, X_train_std, ytr, MODELS, PARAMS, SCORES, ver)

    # evaluation (set plot='yes' so evaluator returns in-memory PIL figures alongside metric dicts)
    best_model_name, best_info, eval_plots_list = evaluator(X_test_processed, X_test_std, yte, models, plot='yes', ver=ver)
    
    test_metrics = best_info['test_metrics']

    result = {
        "version": int(ver),
        "model_name": best_model_name,
        "accuracy": None if pd.isna(test_metrics["accuracy"]) else float(test_metrics["accuracy"]),
        "f1": None if pd.isna(best_info["cv_f1"]) else float(best_info["cv_f1"]),
        "roc_auc": None if pd.isna(test_metrics["roc_auc"]) else float(test_metrics["roc_auc"]),
        "precision": None if pd.isna(test_metrics["precision"]) else float(test_metrics["precision"]),
        "recall": None if pd.isna(test_metrics["recall"]) else float(test_metrics["recall"])
    }

    # if the model's AUC score is below the minimum threshold,
    # reject the model and return the result with a status message.
    if float(result["roc_auc"]) < MIN_AUC:
        print(f"Model rejected: AUC {result['roc_auc']:.3f} < {MIN_AUC}")
        return {"status": "model didn't cross the threshold", **result}

    print(f"Model trained: AUC {result['roc_auc']:.3f} >= {MIN_AUC}")

    # Resolves to project/models/models/v1/active_model.json
    active_path = get_resolved_path(ACTIVE)
    active_path.parent.mkdir(parents=True, exist_ok=True)

    with open(active_path, "w") as f:
        json.dump(result, f, indent=2)

    return {"status": "model is completely trained and functional", **result}


# this function handles the retraining of the model based on new production data.
# it only has retraining logic and does not include the full training pipeline.
# full training logic is handled by the train_pipeline function.
def retrain(ver=1):
    
    # check if the production data file exists and has enough valid labels for retraining.
    prod_p = get_resolved_path(PROD)
    base_p = get_resolved_path(BASE)
    active_p = get_resolved_path(ACTIVE)

    # if the production data file does not exist, return a pending status with a reason.
    if not prod_p.exists():
        return {"status": "pending", "reason": f"Missing file: {prod_p}"}

    prod = pd.read_csv(prod_p).dropna(subset=[TARGET])
    
    # if the production data file does not contain the target column, return a pending status with a reason.
    if TARGET not in prod.columns:
        return {"status": "pending", "reason": "No verified labels"}
    
    # if the number of valid labels in the production data is less than the minimum required for retraining,
    if len(prod) < MIN_LABELS:
        return {"status": "pending", "reason": f"Need {MIN_LABELS} labels; got {len(prod)}"}

    # if the production data file has enough valid labels, proceed with retraining the model.
    cols = [c for c in pd.read_csv(base_p).columns if c != TARGET]
    valid = [c for c in cols + [TARGET] if c in prod.columns]

    old_ver = ver
    
    # if the active model file exists, read the current version from it to determine the next version for retraining.
    if active_p.exists():
        with open(active_p, "r") as f:
            old_ver = json.load(f).get("version", 1)

    # retrain the model using the new production data and increment the version number for the new model.
    result = train_pipeline(ver=old_ver + 1, retrain='yes', data=prod[valid])

    # if the retrained model's AUC score is below the minimum threshold,
    # reject the retrained model and return the result with a status message.
    if result.get("roc_auc", 0) < MIN_AUC:
        return {"status": "rejected", "reason": f"AUC {result.get('roc_auc', 0):.3f} < {MIN_AUC}", **result}

    print(f"New model promoted: AUC {result['roc_auc']:.3f} >= {MIN_AUC}")
    
    return {"status": "promoted", **result}


#prediction pipeline
def run_prediction(data: dict, m_inst, threshold: float = DECISION_THRESHOLD):
    
    model, ver, name = m_inst
    
    x, xs = preprocessor(pd.DataFrame([data]), "no", "yes", ver)
    
    # using data based on the model type (standardized and non-standardized)
    eval_data = xs if any(k in name.lower() for k in ["logistic", "kneighbor", "knc"]) else x
    
    prob = float(model.predict_proba(eval_data)[0, 1])
    
    # finding out the percentage i.e. how high risk they are
    prob_percentage = round(prob * 100, 2)
    
    # prediction based on the threshold
    p = int(prob >= threshold)
    
    # bifurcating customers based on riskyness percentage calculated above
    risk_level = "High Risk" if prob_percentage >= 80.0 else ("Medium Risk" if prob_percentage >= 50.0 else "Low Risk")
    
    # saving the production data
    dt = pd.Timestamp.now().strftime("%Y-%m-%d")
    filepath = PROJECT_ROOT / f"data/new/v{ver}/production_data_{dt}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    
    base_row = {
        "request_id": str(uuid.uuid4()),
        "timestamp": pd.Timestamp.now(),
        "prediction": p,
        "probability": prob,
        "risk_percentage": f"{prob_percentage}%",
        "risk_level": risk_level,
        "threshold": threshold
    }
    
    # putting values in base_row (for frontend)
    for k, val in data.items():
        base_row[k.lower()] = val
    
    # converting base_row to dataframe and saving it
    df_row = pd.DataFrame([base_row])
    
    if filepath.exists(): 
        df_row = df_row.reindex(columns=pd.read_csv(filepath, nrows=0).columns, fill_value="")
        
    df_row.to_csv(filepath, mode="a", header=not filepath.exists(), index=False, quoting=1)
    
    # here base_row itself can be sent as this function's return
    return {
        "request_id": base_row["request_id"],
        "prediction": p,
        "probability": prob,
        "risk_percentage": f"{prob_percentage}%",
        "risk_level": risk_level,
        "threshold_used": threshold,
        "model_version": ver,
        "model": name
    }