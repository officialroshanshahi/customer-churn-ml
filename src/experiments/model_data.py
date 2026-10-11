import os
from src.v1.preprocessing import preprocessor, y_preprocessor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyarrow

ver = 1

#EXTRA CODE FOR EXPORTING DATASETS AND CHECKING CORRELATION
def export_processed_dataset(ver=1):
    #reading the raw data
    #process_data_std, process_data = preprocessor(data_full)
    data_1 = pd.read_csv(f"data/raw/customer_churn_dataset-testing-master.csv")
    data_2 = pd.read_csv(f"data/raw/customer_churn_dataset-training-master.csv")

    data_full = pd.concat([data_1, data_2])

    #splitting into x and y
    X = data_full.drop('Churn', axis=1)
    y = data_full['Churn']

    #imputing missing value in y
    y = y_preprocessor(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #preprocessing x_train and x_test
    x_train_processed, x_train_processed_std = preprocessor(X_train, fit_transform='yes', std='yes', ver=ver)
    x_test_processed, x_test_processed_std = preprocessor(X_test, fit_transform='no', std='yes', ver=ver)


    #exporting them
    os.makedirs(f"data/processed/v{ver}", exist_ok=True)
    
    x_train_processed.to_parquet(f"data/processed/v{ver}/x_train_processed.parquet", index=False)
    x_train_processed_std.to_parquet(f"data/processed/v{ver}/x_train_processed_std.parquet", index=False)
    x_test_processed.to_parquet(f"data/processed/v{ver}/x_test_processed.parquet", index=False)
    x_test_processed_std.to_parquet(f"data/processed/v{ver}/x_test_processed_std.parquet", index=False)
    y_train.to_parquet(f"data/processed/v{ver}/y_train.parquet", index=False)
    y_test.to_parquet(f"data/processed/v{ver}/y_test.parquet", index=False)

    #for sanity check
    # print(x_train_processed.info())
    # print(x_train_processed_std.info())
    #print(x_test_processed.info())
    #print(x_test_processed_std.info())
    #print(x_train_processed.describe())
    #print(x_train_processed_std.describe())
    #print(x_test_processed.describe())
    #print(x_test_processed_std.describe())

    ax = sns.heatmap(
        x_train_processed.corr(), 
        annot=True, 
        square=True, 
        fmt='.2f',
        linewidths=1, 
        linecolor='gray',
        annot_kws={"size": 8}
    )

    plt.xticks(
        rotation=45, 
        ha='right', 
        rotation_mode='anchor',
        fontsize=10
    )

    plt.yticks(fontsize=10)

    plt.tight_layout()

    os.makedirs(f"plots/features/v{ver}", exist_ok=True)
    
    plt.savefig(f"plots/features/v{ver}/feature_correlation.png", bbox_inches="tight", dpi=300)
    plt.show()

export_processed_dataset(ver=ver)