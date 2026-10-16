# Everything the other scripts need: load the data, split it, create the new features, build the pipeline and compute the metrics
# The other scripts do "from data import ..." so we write this code only once

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline   # same as the sklearn Pipeline but it also accepts resampling steps (SMOTE...)

RANDOM_STATE = 42


def load_data():
    # download the dataset from openml (it is saved on the computer after the first time)
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X = adult.data.copy()
    # the target is written as text ">50K" / "<=50K", we turn it into 1 / 0
    y = (adult.target.astype(str) == ">50K").astype(int)
    y.name = "income_over_50k"
    return X, y


def split_data(X, y):
    # 60% train, 20% validation, 20% test, with stratify so each set keeps the same 24% of >50K
    # first we cut the test set (20%), then 0.25 of the remaining 80% = 20% of the whole data for validation
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, random_state=RANDOM_STATE, stratify=y_rest)
    return X_train, X_val, X_test, y_train, y_val, y_test


def add_new_features(X):
    # fixed rules, nothing is learned from the data, so no leakage
    X = X.copy()

    # remove the 2 columns we don't want (education = same info as education-num, fnlwgt = census weight)
    X = X.drop(columns=["fnlwgt", "education"])

    # new features
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["has_capital"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)
    X["log_capital_gain"] = np.log1p(X["capital-gain"])
    X["is_married"] = X["marital-status"].isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
    X["works_overtime"] = (X["hours-per-week"] > 40).astype(int)

    # the text columns are stored as "category", we turn them into normal text so the imputer can add "Missing"
    for col in X.select_dtypes("category").columns:
        X[col] = X[col].astype(object)

    return X


numeric_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "capital_net", "has_capital", "log_capital_gain", "is_married", "works_overtime"]
categorical_cols = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


def build_pipeline(model, sampler=None):
    # numeric columns: missing values -> median, then scaling
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # text columns: missing values -> "Missing", then one-hot encoding
    # min_frequency=0.005: categories with less than 0.5% of the rows are grouped in one "infrequent" column
    # handle_unknown: if a new category shows up later, it goes in that group instead of crashing
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_steps, numeric_cols),
        ("cat", categorical_steps, categorical_cols),
    ])

    steps = [("new_features", FunctionTransformer(add_new_features)), ("preprocess", preprocess)]
    if sampler is not None:
        steps.append(("sampler", sampler))   # resampling (only used during fit, after the preprocessing)
    steps.append(("model", model))
    return ImbPipeline(steps)


def get_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
