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

    #fallback: any boolean-like or binary column
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

    # Drop columns > missing_thresh 
    miss_frac = df.isnull().mean()
    drop_cols += miss_frac[miss_frac > missing_thresh].index.tolist()

    # Drop columns-constants
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
    #parse datelike columns
    for c in df.columns:
        if df[c].dtype == object:
            #datetime
            parsed = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            if parsed.notna().sum() > 0.4 * len(df):
                df[c + "_year"] = parsed.dt.year.fillna(-1).astype(int)
                df[c + "_month"] = parsed.dt.month.fillna(-1).astype(int)
                df[c + "_day"] = parsed.dt.day.fillna(-1).astype(int)
                df.drop(columns=[c], inplace=True)
                continue
            #numeric conversion
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


def prepare_data(df, balance=True):
    """Preprocess dataset: detect target, clean columns, encode categoricals, and balance classes if needed."""
    target_col = find_target_column(df)
    print(f"Auto-detected target column: '{target_col}'")

    # Drop uninformative columns
    df, dropped = drop_uninformative_columns(df, missing_thresh=0.5)
    if dropped:
        print("Dropped uninformative columns:", dropped)

    # Convert datelike/numeric columns
    df = convert_dates_and_numbers(df)

    # Remove constant cols again
    df = df.loc[:, df.nunique(dropna=False) > 1]

    # Separate X and y
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' disappeared after cleaning.")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Remove other churn-like columns
    churn_keywords = ['churn', 'attrition', 'exited', 'cancel', 'canceled', 'closed','Response','target']
    other_churn_cols = [c for c in X.columns if any(k in c.lower() for k in churn_keywords)]
    if other_churn_cols:
        X = X.drop(columns=other_churn_cols)
        print("Removed other churn-like columns from features:", other_churn_cols)

    # Target cleanup (binary)
    if y.dtype == object or y.dtype.name == 'category':
        y = y.fillna("##MISSING##").astype(str)
        le_target = LabelEncoder().fit(y)
        y_enc = le_target.transform(y)
        if len(le_target.classes_) != 2:
            raise ValueError(f"Target column '{target_col}' has {len(le_target.classes_)} classes. Binary expected.")
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
                raise ValueError("Numeric target is not binary.")

    #Group rare categories (<2%)
    for c in X.select_dtypes(include="object").columns:
        top = X[c].value_counts(normalize=True)
        rare = top[top < 0.02].index
        if len(rare) > 0:
            X[c] = X[c].replace(rare, "Other")

    #Create simple engineered ratios if applicable
    if "MonthlyCharges" in X.columns and "tenure" in X.columns:
        X["MonthlyChargePerTenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)
    if "TotalCharges" in X.columns and "tenure" in X.columns:
        X["TotalChargesPerTenure"] = X["TotalCharges"] / (X["tenure"] + 1)

    # Encode categoricals
    X, le_dict = encode_categoricals(X)

    # Fill missing numeric values
    for c in X.columns:
        if X[c].isnull().any():
            if X[c].dtype.kind in 'biufc':
                X[c] = X[c].fillna(X[c].median())
            else:
                X[c] = X[c].fillna(-1)

    # Optional balancing (upsampling)
    if balance:
     pos = (y == 1).sum()
     neg = (y == 0).sum()
     if pos / len(y) < 0.4:
        print("Applying SMOTE for class balancing...")
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
        print(f"After SMOTE → class balance: {y.value_counts().to_dict()}")
   

    return X, y, le_dict, target_col

# def preprocess_for_prediction(df):
#     """Preprocess input data for prediction (no target column)."""

#     df.columns = [c.strip().replace(" ", "").replace("-", "").lower() for c in df.columns]

#     # ✅ Rename all known variations to the expected training names
#     rename_map = {
#         "gender": "Gender",
#         "seniorcitizen": "Senior Citizen",
#         "tenuremonths": "Tenure Months",
#         "phoneservice": "Phone Service",
#         "multiplelines": "Multiple Lines",
#         "internetservice": "Internet Service",
#         "onlinesecurity": "Online Security",
#         "onlinebackup": "Online Backup",
#         "deviceprotection": "Device Protection",
#         "techsupport": "Tech Support",
#         "streamingtv": "Streaming TV",
#         "streamingmovies": "Streaming Movies",
#         "paperlessbilling": "Paperless Billing",
#         "paymentmethod": "Payment Method",
#         "monthlycharges": "Monthly Charges",
#         "totalcharges": "Total Charges",
#         "latitude": "Latitude",
#         "longitude": "Longitude",
#         "cltv": "CLTV",
#         "churnlabel": "Churn Label",
#         "churnvalue": "Churn Value",
#         "churnreason": "Churn Reason"
#     }

#     df.rename(columns=rename_map, inplace=True)

#     # Drop irrelevant columns not used in prediction
#     df.drop(columns=[c for c in df.columns if 'churn' in c.lower() or 'customerid' in c.lower()], errors='ignore')
#     df, dropped = drop_uninformative_columns(df, missing_thresh=0.5)
#     df = convert_dates_and_numbers(df)
#     df = df.loc[:, df.nunique(dropna=False) > 1]

#     # Encode categorical columns
#     X, _ = encode_categoricals(df)

#     # Fill missing numeric values
#     for c in X.columns:
#         if X[c].isnull().any():
#             if X[c].dtype.kind in 'biufc':
#                 X[c] = X[c].fillna(X[c].median())
#             else:
#                 X[c] = X[c].fillna(-1)

#     # Optional engineered ratios
#     if "MonthlyCharges" in X.columns and "tenure" in X.columns:
#         X["MonthlyChargePerTenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)
#     if "TotalCharges" in X.columns and "tenure" in X.columns:
#         X["TotalChargesPerTenure"] = X["TotalCharges"] / (X["tenure"] + 1)

#     return X




def fit_xgb(X_train, y_train, X_valid=None, y_valid=None, random_state=42):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(1, pos)

    model = xgb.XGBClassifier(
        n_estimators=600,#600
        learning_rate=0.3,
        max_depth=6,#6
        min_child_weight=3, #3
        subsample=0.85,#0.85
        colsample_bytree=0.85,#0.85
        gamma=0.3,#0.3
        reg_lambda=2,#2
        reg_alpha=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=random_state,
        use_label_encoder=False,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)] if X_valid is None else [(X_valid, y_valid)],
        verbose=False
    )
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
        # handle multiclass
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

    if show_plots:
        print("\nGenerating Learning Curve (Underfit/Overfit Visualization)...")
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, val_scores = learning_curve(
            baseline, X_train, y_train, cv=5,
            scoring='accuracy',
            train_sizes=np.linspace(0.1, 1.0, 5),
            random_state=random_state
        )
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy')
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Accuracy')
        plt.title("Learning Curve: Underfitting vs Overfitting")
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    # SHAP plot
    if show_plots:
        explainer = shap.TreeExplainer(baseline)
        shap_exp = explainer(X_train)
        shap_values = shap_exp.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=20, show=True)
        plt.show()

    # SHAP feature elimination
    best_features, scores = shap_recursive_feature_elimination(
        X_train, y_train, X_test, y_test, min_features=min_features
    )

    final_model = fit_xgb(X_train[best_features], y_train)
    print("\nFinal Model Performance (After SHAP-RFS):")
    final_metrics = evaluate_model(final_model, X_test[best_features], y_test, prefix="Final: ")

    #SHAP plots
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


import joblib
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal churn pipeline (SHAP-RFS + XGBoost)")
    parser.add_argument("csv", help="path to CSV dataset")
    parser.add_argument("--min_features", type=int, default=5, help="minimum features to keep in RFS")
    args = parser.parse_args()

    results = run_pipeline(args.csv, min_features=args.min_features)

    X, y, _, _ = prepare_data(pd.read_csv(args.csv))
    xgb_final = fit_xgb(X, y)
    joblib.dump(xgb_final, "xgb_churn_model.joblib")
    joblib.dump(results["best_features"], "best_features.joblib")
    print(f" Model saved successfully as {os.path.abspath('xgb_churn_model.joblib')}")

    print("\nDone.")
