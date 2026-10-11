# Customer churn prediction ML system

An end-to-end Machine Learning pipeline designed to predict customer churn, uncover underlying risk factors using SHAP explainability, monitor data drift, and serve real-time predictions via FastApi/Uvicorn backends and a Gradio web application.

Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

---

## Business objective & impact

Customer acquisition is significantly more expensive than customer retention (often 5x to 25x higher depending on the industry). So, predicting when a customer is going to leave the company is a valuable insight for making business decisions.

### Why this model is critical for business?

* **Proactive interventions**: Instead of reacting after a customer cancels, marketing and sales teams receive early warning signals to target high-risk accounts with personalized retention offers or support outreach (of course higher amount of support could also lead to customer leaving the company, **more on that later**).
* **Optimized marketing spend**: Retention budgets can be allocated efficiently toward customers with high churn probability and high overall business value, maximizing ROI.
* **Identifying friction points**: Beyond raw predictions, model explainability pinpoints *why* specific segments are churning (e.g., high support calls combined with low usage), allowing product and operations teams to resolve systemic operational issues.

---
## Project Architecture & Directory Structure

```text
customer-churn-ml-project/
├── api/                        # Production FastAPI/Uvicorn backend services
│   ├── controller/             # retraining and explain triggers (retrain.py and explain.py)
│   ├── database/               # database connection & schema scripts
│   ├── middleware/             # Authentication & request lifecycle hooks
│   ├── uvicorn_init.py         # ASGI Server entrypoint
│   └── api.py                  # Main API routes & inference endpoints
├── app/                        # User interface layer
│   └── gradio.py               # Interactive Gradio dashboard
├── src/                        # Core ML engineering pipeline
│   ├── experiments/            # Exploratory scripts, conversion, & model prototyping
│   ├── v1/                     # Modular ML codebase
│   │   ├── config.py           # Centralized configuration & parameters
│   │   ├── preprocessing.py    # Pipeline transformers & feature encoders
│   │   ├── train.py            # Model training & hyperparameter search
│   │   ├── evaluate.py         # Metrics computation (Precision, Recall, AUC-ROC, F1, PR-AUC)
│   │   ├── explain.py          # SHAP interpretability module
│   │   └── monitor.py          # Data drift and performance degradation monitoring
│   └── __init__.py
├── requirements.txt            # Environment dependencies
├── pyproject.toml              # Environment dependencies
├── Dockerfile                  # containerization
├── prediction_data.json        # data that can be used for /predict or /predict/batch/ or /explain
├── .gitignore                  # GIT ignore file
└── README.md                   # Project documentation

```

---

## Feature engineering & correlation analysis

### Some important features and their implication
Not all feature are explained below for that, please consult the notebook/eda.ipynb. But features that has clear indication for churn is explained below.

#### Age and churn
![Project Screenshot](plots/features/age.png)

From above it is pretty clear that older people seem to churn more. The plot is pretty clear on the churn. When I did bit more digging (go to eda.ipynb for more info), it seems to be they (older customer cohort) seem to have low retention.

**When age and gender is plotted:**

![Project Screenshot](plots/features/age_gender.png)

There doesn't seem to be any correlation (i.e. difference in the churn for 50+ user cohort between male and female). It seems to be pretty equal. So it is not explainable by age.

I also tried support calls, it seems to be very similar if not same. It simply looks like that this cohort is simply churn (if assuming the data is true and fair). Reasons could be:
* company priotising customers who pays the most (different customer cohort). which make sense, wouldn't you priotise those who spend the most.

Of course, it doesn't mean, it is not possible to optimize churn rate for this cohort. Features importance could be used to know which feature to priotised to optimize this customer cohort.

#### Support calls and churn
![Project Screenshot](plots/features/support_calls.png)

Above plot seem to suggest that higher calls isn't necessarily good. Sweet spot seemt to be at 1 or 2. 3 could be used but it seems like there is double digit decline in customer retention.

#### Total spend and churn
![Project Screenshot](plots/features/total_spend.png)

Those who spent above like roughly 500 seem to be less likely to leave the company so when priotising they are the ones. Use /explain endpoint for finding out which feature explains their retention and make it better.

#### Payment delay and churn
![Project Screenshot](plots/features/payment_delay.png)

Those who pay after 20 days should not be priotized.

#### Contract length and churn
![Project Screenshot](plots/features/contract_length.png)

Data seems to suggest that those whose contract length is monthly has high churn rate. To optimize it, company can offer different offers for customers to change their contract length. Offer could be giving certain amount of discount, etc.

### Now, for engineered features

Following non-linear interaction ratios and composite metrics are created:

* `tenure_per_age`: Standardizes relationship length against overall customer maturity.
* `usage_per_tenure`: Measures normalized usage engagement over time.
* `support_calls_per_tenure`: Quantifies customer distress frequency relative to how long they have been subscribed.
* `support_usage_ratio`: Captures operational friction relative to platform engagement.
* `spend_per_tenure`: Reflects annualized revenue rate per customer.
* `churn_risk_signals` & `customer_friction`: Multi-factor composite indices capturing combined risk indicators (e.g., support calls and recency of last interaction).


#### Correlation matrix analysis

![Project Screenshot](plots/features/feature_correlation.png)

#### Why Highly Correlated Features Were Retained:

As shown in the correlation matrix above:

* `tenure` and `tenure_per_age` exhibit a strong correlation ($r = 0.81$).
* `tenure` and `spend_per_tenure` exhibit a correlation of $r = 0.60$.
* `usage_per_tenure` and `spend_per_tenure` exhibit a strong correlation of $r = 0.73$.
* `support_calls` and `churn_risk_signals` show a correlation of $r = 0.60$.
* `support_calls_per_tenure` and `customer_friction` display a strong correlation of $r = 0.42$.


#### Reason for retention:

1. **Tree-Based Model Resilience**: Gradient boosted decision trees (e.g., XGBoost / LightGBM) handle multicollinearity gracefully by selecting optimal split variables independently at each node without destabilizing coefficient estimates (unlike traditional linear regression). **XGBClassifier is used in production so.**
2. **Non-Linear Signals**: Engineered ratios like `tenure_per_age` capture distinct non-linear interaction dynamics that raw underlying variables (like `tenure` or `age`) cannot express independently.
3. **Domain Interpretability**: Business stakeholders often require visibility into both cumulative raw values (e.g., `total_spend`) and normalized rate values (e.g., `spend_per_tenure`) for intervention strategies.

#### Feature selection strategy (SHAP importance pruning)

During the feature engineering phase, additional prospective features were generated—such as interaction terms combining subscription types, logarithmic usage transformations, and recency decay factors.

**Pruning Decisions:**

* **SHAP (SHapley Additive exPlanations) Evaluation**
* **Elimination of Zero-Value Features**

---

## Model evaluation

### Different models evaluation matrix

| Model          | cv_f1    | accuracy | f1   | roc_auc | precision | recall |
| -------------- | -------- | -------- | ---- | ------- | --------- | ------ |
| XGBClassifier  | 0.94     | 0.93     | 0.94 | 0.95    | 0.89      | 0.99   |

![Project Screenshot](plots/evaluate/v1/metrics_comparison.png)

![Project Screenshot](plots/evaluate/v1/pr_curves.png)

![Project Screenshot](plots/evaluate/v1/roc_curves.png)


---

## SHAP plots for different models

From the following plots, it should be clear why some of the engineered features were dropped.

### XGBClassifier - Used in production

![Project Screenshot](plots/shap/v1/XGBoost_waterfall.png)

![Project Screenshot](plots/shap/v1/XGBoost_beeswarm.png)

![Project Screenshot](plots/shap/v1/XGBoost_importance.png)

### RandomForest

![Project Screenshot](plots/shap/v1/RandomForest_importance.png)

![Project Screenshot](plots/shap/v1/RandomForest_waterfall.png)

![Project Screenshot](plots/shap/v1/RandomForest_beeswarm.png)

### DecisionTreeClassifier

![Project Screenshot](plots/shap/v1/DecisionTree_beeswarm.png)

![Project Screenshot](plots/shap/v1/DecisionTree_importance.png)

![Project Screenshot](plots/shap/v1/DecisionTree_waterfall.png)

---

## Quick start

### 1. Environment setup - Using requirements.txt

```bash
# Clone the repository
git clone https://github.com/officialroshanshahi/customer-churn-ml.git
cd customer-churn-ml

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add .env to your root folder
Go to src/experiments and copy the .env. Make necessary changes and add that to your root project folder.
```

### 1.2 Environment setup - Using pyproject.toml

Users can choose between requirements.txt and pyproject.toml:

```bash
# Clone the repository
git clone https://github.com/officialroshanshahi/customer-churn-ml.git
cd customer-churn-ml

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the project and its dependencies using pyproject.toml
pip install .

# Add .env to your root folder
Go to src/experiments and copy the .env. Make necessary changes and add that to your root project folder.

```

### 2. Launch the FastAPI server

Launch the web interface for single-instance or batch churn prediction:

```bash
python -m api.uvicorn_init

```

### 3. Launch the Interactive Gradio Interface

Launch the web interface for single-instance or batch churn prediction:

```bash
python -m app.gradio.py

```
---

## Quick start - Lauching both api and gradio together side by side in VS code:

### 1. Go to src/experiments
### 2. Copy the tasks.json
### 3. Put that file in .vscode
### 4. Press CTRL + SHIFT + P
### 5. Click on Tasks: Run Task
### 6. Select Full-stack customer churn ML model

### Of course, make sure use your virtual environment, and install necessary modules first before doing this.
---

## Model lifecycle & MLOps practices

1. **Preprocessing (`src/v1/preprocessing.py`)**: Handles categorical encoding, missing value imputation, and feature engineering.


2. **Training & evaluation (`src/v1/train.py`, `src/v1/evaluate.py`)**: Executes cross-validation, evaluates metrics for data (Precision-Recall AUC, F1 Score), and serializes model artifacts.


3. **Explainability (`src/v1/explain.py`)**: Integrates SHAP summary plots and individual force plots to explain prediction drivers to business teams.


4. **Monitoring & Drift (`src/v1/monitor.py`)**: Tracks feature distributions between training baseline and incoming live inference streams to alert on data drift. When monitoring endpoint (/monitor) is run, it will create reports/reports_....html.

IMPORTANT NOTE: When running /monitor please make sure that delete data/new/...csv and data/processed/test_baseline


### Training (/health /predict /predict/batch) and retraining pipeline (/monitor) can be initialised through their respective endpoints.