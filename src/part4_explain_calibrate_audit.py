"""Part 4: SHAP, calibration, decision threshold, final test, fairness and pay-equity audit.

Run from the repository root:  python -m src.part4_explain_calibrate_audit
"""

from src.part3_tuning import *  # noqa: F401,F403  (objects of the previous parts)

# ==============================================================================
# Explainability with SHAP
# ==============================================================================

preprocessing = final_pipe[:-1]          # every step except the model
model = final_pipe.named_steps["model"]
preprocess = final_pipe.named_steps["preprocess"]

shap_sample = X_val.sample(2000, random_state=RANDOM_STATE)
X_ready = preprocessing.transform(shap_sample)

# Readable feature names: remove the "num__" / "cat__" prefixes
feature_names = [name.split("__", 1)[1] for name in preprocess.get_feature_names_out()]

# Values to DISPLAY: original numeric values + 0/1 one-hot columns
numeric_columns = preprocess.transformers_[0][2]
n_numeric = len(numeric_columns)
engineered = final_pipe.named_steps["features"].transform(shap_sample)
display_values = np.hstack([engineered[numeric_columns].to_numpy(dtype=float), X_ready[:, n_numeric:]])

if FINAL_MODEL_NAME == "Logistic Regression":
    explainer = shap.LinearExplainer(model, preprocessing.transform(X_train.sample(1000, random_state=RANDOM_STATE)))
else:
    explainer = shap.TreeExplainer(model)

shap_values = explainer(X_ready)
if shap_values.values.ndim == 3:          # Random Forest returns one set of values per class
    shap_values = shap_values[:, :, 1]

shap_values.feature_names = feature_names
shap_values.data = display_values
print("SHAP values shape:", shap_values.values.shape)

# --- Global importance ---

shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Global importance: mean |SHAP value|")
plt.show()

# slightly darker start than BLUES so that low values stay visible as dots on white
dots_cmap = LinearSegmentedColormap.from_list("agency_blues_dots", ["#A9BFDB", NAVY])
shap.plots.beeswarm(shap_values, max_display=15, show=False, color=dots_cmap)
plt.title("SHAP values per person")
plt.show()

global_importance = (
    pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_names)
    .sort_values(ascending=False)
)
display(global_importance.head(15).round(3))

# --- Explaining individual people ---

raw_proba = final_pipe.predict_proba(shap_sample)[:, 1]
y_sample = y_val.loc[shap_sample.index].to_numpy()
positions = np.arange(len(shap_sample))

cases = {
    "Confident >50K (correct)": positions[y_sample == 1][np.argmax(raw_proba[y_sample == 1])],
    "Confident <=50K (correct)": positions[y_sample == 0][np.argmin(raw_proba[y_sample == 0])],
    "Missed high earner": positions[y_sample == 1][np.argmin(raw_proba[y_sample == 1])],
    "False alarm": positions[y_sample == 0][np.argmax(raw_proba[y_sample == 0])],
}

display(pd.DataFrame({
    "true_label": [y_sample[i] for i in cases.values()],
    "model_probability": [raw_proba[i].round(3) for i in cases.values()],
}, index=cases.keys()))

for title, i in cases.items():
    print(f"\n=== {title} ===")
    display(shap_sample.iloc[[i]])
    shap.plots.waterfall(shap_values[i], max_display=10, show=False)
    plt.title(title)
    plt.show()

# ==============================================================================
# Calibration
# ==============================================================================

def expected_calibration_error(y_true, y_proba, n_bins=10):
    y_true = np.asarray(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_proba, bin_edges[1:-1])   # bin number of each person

    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_ids == b
        if in_bin.any():
            gap = abs(y_true[in_bin].mean() - y_proba[in_bin].mean())
            ece += in_bin.mean() * gap                 # weighted by the share of people in the bin
    return ece


def calibration_summary(name, y_true, y_proba):
    return pd.Series({
        "brier_score": brier_score_loss(y_true, y_proba),
        "ece": expected_calibration_error(y_true, y_proba),
        "mean_predicted_probability": y_proba.mean(),
        "observed_rate": np.mean(y_true),
        "pr_auc": average_precision_score(y_true, y_proba),
    }, name=name)

val_proba_raw = final_pipe.predict_proba(X_val)[:, 1]

fig, ax = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(y_val, val_proba_raw, n_bins=10, name=f"{FINAL_MODEL_NAME} (raw)", ax=ax)
ax.set_title("Reliability curve: validation set")
plt.show()

display(calibration_summary("raw", y_val, val_proba_raw).round(4))

# --- Recalibration ---

calibrated_models = {}
summaries = [calibration_summary("raw", y_val, val_proba_raw)]

for method in ["sigmoid", "isotonic"]:
    calibrated = CalibratedClassifierCV(clone(final_pipe), method=method, cv=5)
    calibrated.fit(X_train, y_train)
    calibrated_models[method] = calibrated
    summaries.append(calibration_summary(method, y_val, calibrated.predict_proba(X_val)[:, 1]))

calibration_table = pd.DataFrame(summaries)
display(calibration_table.round(4))

fig, ax = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(y_val, val_proba_raw, n_bins=10, name="raw", ax=ax)
for method, calibrated in calibrated_models.items():
    CalibrationDisplay.from_predictions(
        y_val, calibrated.predict_proba(X_val)[:, 1], n_bins=10, name=method, ax=ax,
    )
ax.set_title("Reliability curves: before and after calibration (validation)")
plt.show()

CALIBRATION_METHOD = calibration_table["brier_score"].idxmin()
calibrated_model = calibrated_models[CALIBRATION_METHOD]
val_proba = calibrated_model.predict_proba(X_val)[:, 1]

print("Chosen calibration method:", CALIBRATION_METHOD)

# --- How many people support each point of the curve? ---

bins = np.linspace(0, 1, 11)
bin_table = (
    pd.DataFrame({"bin": pd.cut(val_proba, bins, include_lowest=True), "y": y_val.to_numpy(), "p": val_proba})
    .groupby("bin", observed=False)
    .agg(n_people=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(range(len(bin_table)), bin_table["n_people"], color=NAVY)
ax.set_xticks(range(len(bin_table)), [f"{b.left:.1f}-{b.right:.1f}" for b in bin_table.index], rotation=45)
ax.set_xlabel("Calibrated probability (validation)")
ax.set_ylabel("Number of people")
ax.set_title("People per probability bin")
plt.tight_layout()
plt.show()

display(bin_table.round(3))

# ==============================================================================
# Decision rule for individual flags
# ==============================================================================

precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)

best = np.argmax(f1_scores[:-1])           # the last point has no threshold
F1_THRESHOLD = thresholds[best]

THRESHOLD = 0.5                            # "more likely than not" on calibrated probabilities
val_pred = (val_proba >= THRESHOLD).astype(int)

print(f"Threshold maximising F1: {F1_THRESHOLD:.3f} | precision {precisions[best]:.3f} | recall {recalls[best]:.3f} | F1 {f1_scores[best]:.3f}")
print(f"Chosen threshold: {THRESHOLD}        | precision {precision_score(y_val, val_pred):.3f} | recall {recall_score(y_val, val_pred):.3f} | F1 {f1_score(y_val, val_pred):.3f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.plot(thresholds, f1_scores[:-1], label="F1")
plt.axvline(THRESHOLD, linestyle="--", color=INK, label=f"Chosen threshold = {THRESHOLD}")
plt.axvline(F1_THRESHOLD, linestyle=":", color=STEEL, label=f"F1-maximising threshold = {F1_THRESHOLD:.2f}")
plt.xlabel("Decision threshold (calibrated probability)")
plt.ylabel("Score (validation)")
plt.title("Precision, recall and F1 versus the threshold")
plt.legend()
plt.show()

# ==============================================================================
# Final evaluation on the test set
# ==============================================================================

test_proba = calibrated_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= THRESHOLD).astype(int)

val_pred = (val_proba >= THRESHOLD).astype(int)

final_results = pd.DataFrame({
    "validation": classification_metrics(y_val, val_pred, val_proba),
    "test": classification_metrics(y_test, test_pred, test_proba),
})
final_results.loc["brier_score"] = [brier_score_loss(y_val, val_proba), brier_score_loss(y_test, test_proba)]
final_results.loc["ece"] = [expected_calibration_error(y_val, val_proba), expected_calibration_error(y_test, test_proba)]
display(final_results.round(4))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, test_pred, display_labels=["<=50K", ">50K"], ax=axes[0], colorbar=False, cmap=BLUES,
)
axes[0].set_title("Confusion matrix (test)")

RocCurveDisplay.from_predictions(y_test, test_proba, name=FINAL_MODEL_NAME, ax=axes[1])
axes[1].set_title("ROC curve (test)")

PrecisionRecallDisplay.from_predictions(y_test, test_proba, name=FINAL_MODEL_NAME, ax=axes[2])
axes[2].set_title("Precision-recall curve (test)")

plt.tight_layout()
plt.show()

# ==============================================================================
# Fairness audit
# ==============================================================================
# --- Classic fairness metrics ---

fairness_metrics = {
    "count": lambda y_true, y_pred: len(y_true),
    "base_rate": lambda y_true, y_pred: np.mean(y_true),     # actual share earning >50K
    "selection_rate": selection_rate,                        # share predicted >50K
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate,              # = expected >50K but paid <=50K
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
}


def fairness_report(y_true, y_pred, sensitive):
    frame = MetricFrame(metrics=fairness_metrics, y_true=y_true, y_pred=y_pred,
                        sensitive_features=sensitive)
    gaps = pd.Series({
        "demographic_parity_difference": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive),
        "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
    })
    return frame, gaps


frame_sex, gaps_sex = fairness_report(y_test, test_pred, X_test["sex"])
display(frame_sex.by_group.round(3))
display(gaps_sex.round(3))

frame_race, gaps_race = fairness_report(y_test, test_pred, X_test["race"])
display(frame_race.by_group.round(3))
display(gaps_race.round(3))

# --- Actual vs expected: the pay-equity audit ---

def bootstrap_gap_ci(actual, expected, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(actual)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)            # resample the group's people with replacement
        gaps.append(actual[idx].mean() - expected[idx].mean())
    return np.percentile(gaps, [2.5, 97.5])


def pay_equity_audit(y_true, y_proba, groups, threshold=THRESHOLD, min_size=100):
    data = pd.DataFrame({
        "group": np.asarray(groups).astype(str),
        "actual": np.asarray(y_true),
        "expected": np.asarray(y_proba),
    })
    rows = []
    for group, part in data.groupby("group"):
        actual, expected = part["actual"].to_numpy(), part["expected"].to_numpy()
        low, high = bootstrap_gap_ci(actual, expected)
        expected_high = expected >= threshold
        rows.append({
            "group": group,
            "n": len(part),
            "actual_rate": actual.mean(),
            "expected_rate": expected.mean(),
            "unexplained_gap_pts": 100 * (actual.mean() - expected.mean()),
            "ci_low_pts": 100 * low,
            "ci_high_pts": 100 * high,
            "underpaid_rate": (actual[expected_high] == 0).mean() if expected_high.any() else np.nan,
            "enough_data": len(part) >= min_size,
        })
    return pd.DataFrame(rows).set_index("group").sort_values("unexplained_gap_pts")

audit_sex = pay_equity_audit(y_test, test_proba, X_test["sex"])
display(audit_sex.round(3))

audit_race = pay_equity_audit(y_test, test_proba, X_test["race"])
display(audit_race.round(3))

fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 2]})

for ax, (title, table) in zip(axes, [("Sex", audit_sex), ("Race", audit_race)]):
    table[["actual_rate", "expected_rate"]].plot(kind="bar", ax=ax)
    ax.set_title(f"Actual vs expected share earning >50K: {title} (test set)")
    ax.set_ylabel("Share earning >50K")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# Unexplained gap with its 95% confidence interval (groups with enough data)
audit_all = pd.concat([audit_sex.assign(attribute="sex"), audit_race.assign(attribute="race")])
shown = audit_all[audit_all["enough_data"]].sort_values("unexplained_gap_pts")

plt.figure(figsize=(8, 4))
plt.errorbar(
    shown["unexplained_gap_pts"], range(len(shown)), color=NAVY,
    xerr=[shown["unexplained_gap_pts"] - shown["ci_low_pts"], shown["ci_high_pts"] - shown["unexplained_gap_pts"]],
    fmt="o", capsize=4,
)
plt.yticks(range(len(shown)), shown.index)
plt.axvline(0, color=STEEL, linestyle="--")
plt.xlabel("Unexplained gap (percentage points), 95% CI")
plt.title("Actual minus expected share earning >50K (test set)")
plt.show()

# --- Mitigation: before / after on the test set ---

naive_pipe = clone(final_pipe).set_params(features__drop_columns=NOT_USEFUL_COLUMNS)

naive_model = CalibratedClassifierCV(naive_pipe, method=CALIBRATION_METHOD, cv=5)
naive_model.fit(X_train, y_train)

naive_proba = naive_model.predict_proba(X_test)[:, 1]
naive_pred = (naive_proba >= THRESHOLD).astype(int)

naive_audit_sex = pay_equity_audit(y_test, naive_proba, X_test["sex"])
naive_audit_race = pay_equity_audit(y_test, naive_proba, X_test["race"])
display(naive_audit_sex.round(3))

def summary_row(y_proba, y_pred, audit_sex, audit_race):
    _, sex_gaps = fairness_report(y_test, y_pred, X_test["sex"])
    return pd.Series({
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "demographic_parity_difference_sex": sex_gaps["demographic_parity_difference"],
        "equalized_odds_difference_sex": sex_gaps["equalized_odds_difference"],
        "unexplained_gap_women_pts": audit_sex.loc["Female", "unexplained_gap_pts"],
        "unexplained_gap_men_pts": audit_sex.loc["Male", "unexplained_gap_pts"],
        "unexplained_gap_black_pts": audit_race.loc["Black", "unexplained_gap_pts"],
    })


before_after = pd.DataFrame({
    "before: all columns": summary_row(naive_proba, naive_pred, naive_audit_sex, naive_audit_race),
    "after: job-related only": summary_row(test_proba, test_pred, audit_sex, audit_race),
})
display(before_after.round(3))

fig, ax = plt.subplots(figsize=(9, 4.5))
pd.DataFrame({
    "before: all columns": naive_audit_sex["unexplained_gap_pts"],
    "after: job-related only": audit_sex["unexplained_gap_pts"],
}).plot(kind="bar", ax=ax, color=[SILVER, NAVY])      # grey = before (reference), navy = our model
ax.axhline(0, color=STEEL)
ax.set_ylabel("Unexplained gap (percentage points)")
ax.set_title("What the audit finds, depending on the model (test set)")
ax.tick_params(axis="x", rotation=0)
plt.show()

# --- Why not a post-processing method such as ThresholdOptimizer? ---
# --- Proxy check: do the inputs still "know" a person's sex? ---

model_inputs = final_pipe[:-1].transform(X_train)          # the exact inputs of our model
is_female = (X_train["sex"] == "Female").astype(int)

proxy_auc = cross_val_score(
    LogisticRegression(max_iter=2000), model_inputs, is_female, cv=5, scoring="roc_auc",
).mean()
print(f"ROC-AUC for predicting sex from the model inputs: {proxy_auc:.3f}")

# --- Sensitivity check: does the audit depend on the capital columns? ---

CAPITAL_COLUMNS = ["capital-gain", "capital-loss"]


class DropCapitalFeatures(BaseEstimator, TransformerMixin):
    # Removes the capital columns and the features engineered from them
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        engineered = ["capital_net", "has_capital", "log_capital_gain"]
        return X.drop(columns=[c for c in CAPITAL_COLUMNS + engineered if c in X.columns])


no_capital_pipe = clone(final_pipe)
no_capital_pipe.steps.insert(1, ("drop_capital", DropCapitalFeatures()))   # right after feature engineering

no_capital_model = CalibratedClassifierCV(no_capital_pipe, method=CALIBRATION_METHOD, cv=5)
no_capital_model.fit(X_train, y_train)
no_capital_proba = no_capital_model.predict_proba(X_test)[:, 1]

print(f"Test ROC-AUC without capital columns: {roc_auc_score(y_test, no_capital_proba):.3f} "
      f"(with: {roc_auc_score(y_test, test_proba):.3f})")

sensitivity = pd.concat({
    "with capital columns": pd.concat([audit_sex, audit_race])[["unexplained_gap_pts", "ci_low_pts", "ci_high_pts"]],
    "without capital columns": pd.concat([
        pay_equity_audit(y_test, no_capital_proba, X_test["sex"]),
        pay_equity_audit(y_test, no_capital_proba, X_test["race"]),
    ])[["unexplained_gap_pts", "ci_low_pts", "ci_high_pts"]],
}, axis=1)
display(sensitivity.loc[["Female", "Male", "Black", "White", "Asian-Pac-Islander"]].round(2))

# --- Log the final evaluation and the audit in MLflow ---

with mlflow.start_run(run_name=f"{FINAL_MODEL_NAME}: final test & audit"):
    mlflow.set_tags({"stage": "final_test", "model_type": FINAL_MODEL_NAME, "source_run_id": final_run_id})
    mlflow.log_params({
        "calibration_method": CALIBRATION_METHOD,
        "threshold": THRESHOLD,
        "excluded_columns": ", ".join(REFERENCE_DROP),
        "fairness_mitigation": "pre-processing: protected attributes and proxies removed",
    })
    mlflow.log_metrics({f"test_{k}": v for k, v in final_results["test"].items()})
    mlflow.log_metrics({f"test_{k}_sex": v for k, v in gaps_sex.items()})
    mlflow.log_metrics({
        "no_capital_test_roc_auc": roc_auc_score(y_test, no_capital_proba),
        "unexplained_gap_women_pts": audit_sex.loc["Female", "unexplained_gap_pts"],
        "unexplained_gap_men_pts": audit_sex.loc["Male", "unexplained_gap_pts"],
        "unexplained_gap_black_pts": audit_race.loc["Black", "unexplained_gap_pts"],
        "naive_unexplained_gap_women_pts": naive_audit_sex.loc["Female", "unexplained_gap_pts"],
        "proxy_auc_sex": proxy_auc,
    })
    mlflow.log_table(audit_sex.reset_index(), "audit_by_sex.json")
    mlflow.log_table(audit_race.reset_index(), "audit_by_race.json")
    mlflow.log_table(before_after.reset_index(names="metric"), "mitigation_before_after.json")
    mlflow.log_table(global_importance.reset_index(name="mean_abs_shap").rename(columns={"index": "feature"}),
                     "shap_global_importance.json")
    mlflow.sklearn.log_model(calibrated_model, name="calibrated_model", serialization_format="cloudpickle")

print("Final evaluation and audit logged.")

if DRIVE_BACKUP_DIR is not None:
    shutil.copytree(MLFLOW_DIR, DRIVE_BACKUP_DIR, dirs_exist_ok=True)
    print("MLflow data copied to", DRIVE_BACKUP_DIR)
elif IN_COLAB:
    # No Drive: zip the runs and download them to your computer instead
    from google.colab import files
    archive = shutil.make_archive("/content/mlflow_backup", "zip", MLFLOW_DIR)
    files.download(archive)
else:
    print("Not in Colab: MLflow data is already saved in", MLFLOW_DIR)
