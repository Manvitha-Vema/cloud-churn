import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                        f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb
import shap
from copy import deepcopy


# Helper utilities

def find_target_column(df):
    """Find a column that likely represents churn/attrition."""
    keywords = ['churn', 'attrition', 'exited', 'cancel', 'canceled', 'closed','target']
    candidates = []
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords):
            candidates.append(c)
    # fallback: any boolean-like or binary column
    if not candidates:
        for c in df.columns:
            if df[c].dropna().isin([0,1,'0','1',True,False,'True','False']).all():
                candidates.append(c)
    if not candidates:
        raise ValueError("No churn-like column found. Rename your churn column (contains 'churn' or similar).")
    # prefer exact 'churn' or first candidate
    for cand in candidates:
        if cand.lower() == 'churn' or 'churn value' in cand.lower() or 'churn_label' in cand.lower():
            return cand
    return candidates[0]

def drop_uninformative_columns(df, missing_thresh=0.5):
    n = len(df)
    drop_cols = []

    # Drop columns with > missing_thresh fraction missing
    miss_frac = df.isnull().mean()
    drop_cols += miss_frac[miss_frac > missing_thresh].index.tolist()

    # Drop columns that are constants
    nunique = df.nunique(dropna=False)
    drop_cols += nunique[nunique <= 1].index.tolist()

    # Drop ID-like columns: unique per row or explicit 'id' in name
    for c in df.columns:
        if df[c].nunique(dropna=False) == n:
            drop_cols.append(c)
        if 'id' in c.lower() and df[c].nunique() > 10:
            drop_cols.append(c)

    # Remove duplicates and return unique list
    drop_cols = list(set(drop_cols))
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df, drop_cols

def convert_dates_and_numbers(df):
    # Try to parse datelike columns
    for c in df.columns:
        if df[c].dtype == object:
            # try datetime
            parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            if parsed.notna().sum() > 0.4 * len(df):  # fairly datelike
                df[c + "_year"] = parsed.dt.year.fillna(-1).astype(int)
                df[c + "_month"] = parsed.dt.month.fillna(-1).astype(int)
                df[c + "_day"] = parsed.dt.day.fillna(-1).astype(int)
                df.drop(columns=[c], inplace=True)
                continue
            # try numeric conversion
            numconv = pd.to_numeric(df[c].str.replace(',',''), errors='coerce')
            if numconv.notna().sum() > 0.5 * len(df):
                df[c] = numconv
    return df

def encode_categoricals(df, exclude_cols=None):
    """Label-encode object columns. Returns encoder dict for reuse."""
    if exclude_cols is None: exclude_cols = []
    le_dict = {}
    for c in df.select_dtypes(include=['object', 'category']).columns:
        if c in exclude_cols: 
            continue
        df[c] = df[c].fillna("##MISSING##").astype(str)
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        le_dict[c] = le
    return df, le_dict


def prepare_data(df):
    # 1) find target
    target_col = find_target_column(df)
    print(f"Auto-detected target column: '{target_col}'")

    # 2) drop uninformative columns
    df, dropped = drop_uninformative_columns(df, missing_thresh=0.5)
    if dropped:
        print("Dropped uninformative columns:", dropped)

    # 3) convert numerics and datetimes
    df = convert_dates_and_numbers(df)

    # 4) remove constant cols again
    df = df.loc[:, df.nunique(dropna=False) > 1]

    # 5) separate X and y
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' disappeared after cleaning.")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 6) Remove any other churn/attrition-like columns from features
    churn_keywords = ['churn', 'attrition', 'exited', 'cancel', 'canceled', 'closed']
    other_churn_cols = [c for c in X.columns if any(k in c.lower() for k in churn_keywords)]
    if other_churn_cols:
        X = X.drop(columns=other_churn_cols)
        print("Removed other churn-like columns from features:", other_churn_cols)

    # 7) target clean-up: binary encoding
    if y.dtype == object or y.dtype.name == 'category':
        y = y.fillna("##MISSING##").astype(str)
        le_target = LabelEncoder().fit(y)
        y_enc = le_target.transform(y)
        if len(le_target.classes_) != 2:
            raise ValueError(f"Target column '{target_col}' has {len(le_target.classes_)} classes. This pipeline expects binary churn.")
        y = pd.Series(y_enc, index=y.index, name=target_col)
        print("Target classes (encoded):", dict(enumerate(le_target.classes_)))
    else:
        uniq = pd.Series(y.dropna().unique())
        if set(uniq.astype(int).tolist()) <= {0,1} and len(uniq) <= 2:
            y = y.astype(int)
        else:
            if y.nunique() == 2:
                y = y.map({y.unique()[0]:0, y.unique()[1]:1})
            else:
                raise ValueError("Numeric target is not binary. Pipeline expects binary churn target.")

    # 8) encode categorical predictors
    X, le_dict = encode_categoricals(X)

    # 9) fill remaining numeric NaNs
    for c in X.columns:
        if X[c].isnull().any():
            if X[c].dtype.kind in 'biufc':
                X[c] = X[c].fillna(X[c].median())
            else:
                X[c] = X[c].fillna(-1)

    return X, y, le_dict, target_col

# ---------------------------
# Modeling + SHAP-RFS
# ---------------------------
def fit_xgb(X_train, y_train, X_valid=None, y_valid=None, random_state=42, n_estimators=200, max_depth=5, lr=0.1, subsample=0.8, colsample=0.8):
    # compute scale_pos_weight for imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = 1.0
    if pos > 0:
        scale_pos_weight = neg / max(1, pos)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test, prefix=""):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    roc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)
    print(f"{prefix}Accuracy : {acc:.4f}")
    print(f"{prefix}Precision: {prec:.4f}")
    print(f"{prefix}Recall   : {rec:.4f}")
    print(f"{prefix}F1-Score : {f1:.4f}")
    print(f"{prefix}ROC-AUC  : {roc:.4f}")
    print(f"{prefix}Confusion Matrix:\n{cm}")
    return {'acc':acc,'prec':prec,'rec':rec,'f1':f1,'roc':roc,'cm':cm}

def shap_recursive_feature_elimination(X_train, y_train, X_test, y_test,
                                       min_features=5, performance_tol=0.005):
    Xtr = X_train.copy()
    Xte = X_test.copy()
    current_features = Xtr.columns.tolist()
    best_score = 0
    best_features = current_features.copy()
    scores = []
    iteration = 0

    while len(current_features) > min_features:
        iteration += 1
        model = fit_xgb(Xtr[current_features], y_train)
        y_prob = model.predict_proba(Xte[current_features])[:,1]
        roc = roc_auc_score(y_test, y_prob)
        scores.append((len(current_features), roc))

        if roc >= best_score - performance_tol:
            best_score = roc
            best_features = current_features.copy()

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(Xtr[current_features])
        shap_vals = shap_exp.values
        # handle multiclass like before
        if shap_vals.ndim == 3:
            # choose class 1
            shap_vals = shap_vals[:, :, 1]
        shap_mean = np.abs(shap_vals).mean(axis=0)
        # remove the lowest impact feature
        num_to_remove = 1
        idx_to_remove = np.argsort(shap_mean)[:num_to_remove]
        feats_to_remove = [current_features[i] for i in idx_to_remove]
        print(f"Iteration {iteration}: Removing {feats_to_remove}, ROC-AUC={roc:.4f}")
        for f in feats_to_remove:
            current_features.remove(f)

    print("\nRFS complete.")
    print("Best feature subset (SHAP-RFS):")
    print(best_features)
    print(f"Best ROC-AUC achieved: {best_score:.4f}")
    return best_features, scores

# ---------------------------
# Main pipeline
# # ---------------------------
# def run_pipeline(csv_path, test_size=0.2, random_state=42, min_features=5):
#     df = pd.read_csv(csv_path)
#     print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
# def run_pipeline(data, test_size=0.2, random_state=42, min_features=5):
#     """
#     Accepts:
#     - data: CSV file path OR pandas DataFrame
#     """
#     if isinstance(data, str):
#         df = pd.read_csv(data)
#     elif isinstance(data, pd.DataFrame):
#         df = data.copy()
#     else:
#         raise ValueError("Input must be a CSV path or pandas DataFrame")

#     print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

#     # prepare
#     X, y, le_dict, target_col = prepare_data(df)

#     # split with safe stratify
#     stratify = y if (y.value_counts(normalize=True).min() > 0.05 and y.nunique() == 2) else None
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
#     print("Preprocessing completed! Feature shape:", X_train.shape)

#     # baseline model
#     baseline = fit_xgb(X_train, y_train)
#     print("\nBaseline Model Performance (Before SHAP-RFS):")
#     baseline_metrics = evaluate_model(baseline, X_test, y_test, prefix="Baseline: ")

#     # SHAP on baseline (bar)
#     explainer = shap.TreeExplainer(baseline)
#     shap_exp = explainer(X_train)
#     shap_values = shap_exp.values
#     if shap_values.ndim == 3:
#         shap_values = shap_values[:,:,1]
#     plt.figure(figsize=(10,6))
#     shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=20, show=True)
#     plt.show()

#     # SHAP-guided RFS
#     best_features, scores = shap_recursive_feature_elimination(X_train, y_train, X_test, y_test, min_features=min_features)

#     # final model on best features
#     final_model = fit_xgb(X_train[best_features], y_train)
#     print("\nFinal Model Performance (After SHAP-RFS):")
#     final_metrics = evaluate_model(final_model, X_test[best_features], y_test, prefix="Final: ")

#     # SHAP for final model (dot + bar)
#     explainer = shap.TreeExplainer(final_model)
#     shap_exp = explainer(X_train[best_features])
#     shap_vals = shap_exp.values
#     if shap_vals.ndim == 3:
#         shap_vals = shap_vals[:,:,1]

#     print("SHAP values shape:", shap_vals.shape)
#     print("Training data shape:", X_train[best_features].shape)

#     plt.figure(figsize=(10,6))
#     shap.summary_plot(shap_vals, X_train[best_features], plot_type="bar", max_display=20, show=True)
#     plt.show()

#     plt.figure(figsize=(10,6))
#     shap.summary_plot(shap_vals, X_train[best_features], plot_type="dot", max_display=20, show=True)
#     plt.show()

#     return {
#         'baseline_metrics': baseline_metrics,
#         'final_metrics': final_metrics,
#         'best_features': best_features,
#         'scores': scores,
#         'le_dict': le_dict,
#         'target_col': target_col
#     }
def run_pipeline(data, test_size=0.2, random_state=42, min_features=5, show_plots=True):
    """
    Accepts:
    - data: CSV file path OR pandas DataFrame
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a CSV path or pandas DataFrame")

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if len(df) < 5:
      raise ValueError("Please provide at least a few rows of data for model training/testing.")


    # prepare
    X, y, le_dict, target_col = prepare_data(df)

    stratify = y if (y.value_counts(normalize=True).min() > 0.05 and y.nunique() == 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print("Preprocessing completed! Feature shape:", X_train.shape)

    # baseline model
    baseline = fit_xgb(X_train, y_train)
    print("\nBaseline Model Performance (Before SHAP-RFS):")
    baseline_metrics = evaluate_model(baseline, X_test, y_test, prefix="Baseline: ")

    # SHAP baseline plot
    if show_plots:
        explainer = shap.TreeExplainer(baseline)
        shap_exp = explainer(X_train)
        shap_values = shap_exp.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=20, show=True)
        plt.show()

    # SHAP-guided feature elimination
    best_features, scores = shap_recursive_feature_elimination(
        X_train, y_train, X_test, y_test, min_features=min_features
    )

    final_model = fit_xgb(X_train[best_features], y_train)
    print("\nFinal Model Performance (After SHAP-RFS):")
    final_metrics = evaluate_model(final_model, X_test[best_features], y_test, prefix="Final: ")

    # Final SHAP plots
    if show_plots:
        explainer = shap.TreeExplainer(final_model)
        shap_exp = explainer(X_train[best_features])
        shap_vals = shap_exp.values
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_train[best_features], plot_type="bar", max_display=20, show=True)
        plt.show()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_train[best_features], plot_type="dot", max_display=20, show=True)
        plt.show()

    return {
        'baseline_metrics': baseline_metrics,
        'final_metrics': final_metrics,
        'best_features': best_features,
        'scores': scores,
        'le_dict': le_dict,
        'target_col': target_col
    }


# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal churn pipeline (SHAP-RFS + XGBoost)")
    parser.add_argument("csv", help="path to CSV dataset")
    parser.add_argument("--min_features", type=int, default=5, help="minimum features to keep in RFS")
    args = parser.parse_args()

    results = run_pipeline(args.csv, min_features=args.min_features)
    print("\nDone.")
