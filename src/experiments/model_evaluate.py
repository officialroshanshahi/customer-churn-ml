import joblib
import pandas as pd

#EXTRA CODE
ver = 1


import datetime
import io
import joblib
import pandas as pd
import matplotlib

# set the backend to 'Agg' for non-interactive environments (like servers or CI/CD pipelines)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, precision_recall_curve, 
    recall_score, roc_curve
)
from pathlib import Path

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# helper function to generate evaluation plots and save them to disk
def eval_plots(best_models, ver=1):
    
    print("\nGenerating evaluation plots...")

    rows = []
    
    # iterate over the best models and their evaluation metrics to prepare a DataFrame for plotting
    for name, r in best_models.items():
        rows.append({
            "model": name,
            "cv_f1": r["cv_f1"],
            **r["test_metrics"]
        })

    # create a DataFrame from the collected metrics for easier plotting and saving
    df = pd.DataFrame(rows)
    
    # create a directory for evaluation plots if it doesn't exist
    eval_dir = PROJECT_ROOT / f"plots/evaluate/v{ver}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation metrics to CSV and Excel files with timestamped filenames
    df.to_csv(eval_dir / f"model_metrics_v{ver}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)
    df.to_excel(eval_dir / f"model_metrics_v{ver}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx", index=False)

    generated_images = []

    # helper function to convert matplotlib figures to PIL images for in-memory storage and later use
    def fig_to_pil():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    # metrics comparison plot
    plt.figure(figsize=(10, 5))
    df.set_index("model")[["accuracy", "f1", "roc_auc", "precision", "recall"]].plot(
        kind="bar", figsize=(10, 5)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    
    # save the metrics comparison plot to the generated images list
    generated_images.append(fig_to_pil())

    # ROC curve
    plt.figure(figsize=(8, 5))
    for name, r in best_models.items():
        fpr, tpr = r["curves"]["roc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={r['test_metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    
    generated_images.append(fig_to_pil())

    # Precision-Recall curve
    plt.figure(figsize=(8, 5))
    for name, r in best_models.items():
        precision, recall = r["curves"]["pr"]
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    
    # fig_to_pil() converts the current matplotlib figure to a PIL image and appends it to the generated_images list
    generated_images.append(fig_to_pil())

    print(f"Results saved to: {eval_dir}/")
    
    # print the DataFrame with rounded values for better readability in the console
    # this can be removed or commented out if not needed, as the CSV and Excel files already contain the metrics
    print(df.round(4))
    
    # return the list of generated PIL images for further use or display
    return generated_images


# evaluator function to assess the performance of trained models on test data
# it calculates various metrics, generates evaluation plots, and saves the best model based on cross-validation F1 score.
# the function returns the name of the best model, its evaluation metrics, and a list of generated plots (if requested).
# saves the evaluation metrics and plots to disk for future reference and analysis.
def evaluator(x_test_processed, x_test_processed_std, y_test, trained_models, cv_scores, plot='no', ver=1):
    
    best_models = {}

    
    print("\nEvaluating models on test data...")

    # iterate over each trained model to calculate predictions, probabilities, and evaluation metrics
    for model_name, model in trained_models.items():
        
        m_lower = model_name.lower()
        
        # determine which test data to use based on the model type (standardized or not)
        if "logistic" in m_lower or "kneighbor" in m_lower:
            X_te = x_test_processed_std
        else:
            X_te = x_test_processed

        y_pred = model.predict(X_te)

        # check if the model has a predict_proba method (for probability predictions)
        # or use decision_function for models that don't
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_te)[:, 1]
        else:
            y_probs = model.decision_function(X_te)
        
        # calculate evaluation metrics and store them in the best_models dictionary for later comparison and plotting
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)

        # store the evaluation metrics and curves for each model in the best_models dictionary
        best_models[model_name] = {
            'model': model,
            'cv_f1': cv_scores.get(model_name, 0.0),
            'test_metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_probs),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            },
            'curves': {
                'roc': (fpr, tpr),
                'pr': (precision, recall)
            }
        }


    # find best model based on cv_f1 score
    best_model_name, _ = max(
        best_models.items(), 
        key=lambda item: item[1]['cv_f1']
    )

    generated_plots = []
    
    # if the user requested plots, generate them and store them in the generated_plots list
    if str(plot).lower() == "yes":
        generated_plots = eval_plots(best_models, ver=ver)
        
    return generated_plots


def evaluate_models(ver=1):
    x_test_processed = pd.read_parquet(f'data/processed/v{ver}/x_test_processed.parquet')
    x_test_processed_std = pd.read_parquet(f'data/processed/v{ver}/x_test_processed_std.parquet')
    y_test = pd.read_parquet(f'data/processed/v{ver}/y_test.parquet')

    searches = {
        'DecisionTree': joblib.load(f'models/ev_search/v{ver}/DecisionTreeClassifier_search_{ver}.pkl'),
        'LogisticRegression': joblib.load(f'models/ev_search/v{ver}/LogisticRegression_search_{ver}.pkl'),
        'RandomForest': joblib.load(f'models/ev_search/v{ver}/RandomForest_search_{ver}.pkl'),
        'XGBClassifier': joblib.load(f'models/ev_search/v{ver}/XGBClassifier_search_{ver}.pkl')
    }

    cv_scores = {
        name: search.best_score_
        for name, search in searches.items()
    }

    best_params = {
        name: search.best_params_
        for name, search in searches.items()
    }


    trained_models = {
        name: search.best_estimator_
        for name, search in searches.items()
    }

    best_model = evaluator(x_test_processed, x_test_processed_std, y_test, trained_models, cv_scores, plot='yes', ver=ver)

evaluate_models(ver=ver)
