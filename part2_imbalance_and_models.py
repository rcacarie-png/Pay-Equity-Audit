"""Part 2 — MLflow tracking, class-imbalance remedies and comparison of three models.

Run from the repository root:  python -m src.part2_imbalance_and_models
"""

from src.part1_data_features_baseline import *  # noqa: F401,F403  (objects of the previous parts)

# ==============================================================================
# Experiment tracking with MLflow
# ==============================================================================

try:
    import google.colab  # only exists when running on Colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DRIVE_BACKUP_DIR = None  # stays None if Drive is not available

if IN_COLAB:
    MLFLOW_DIR = "/content/mlflow"
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Supervised Learning & Model Engineering/Final Project/Code adapted to problematic/mlflow"
        if os.path.exists(DRIVE_BACKUP_DIR):            # restore previous runs
            shutil.copytree(DRIVE_BACKUP_DIR, MLFLOW_DIR, dirs_exist_ok=True)
    except Exception as error:
        # If Drive cannot be mounted, MLflow still works locally for this session
        print("Could not mount Google Drive:", error)
        print("MLflow will work, but download the runs at the end (Part V).")
else:
    MLFLOW_DIR = os.path.abspath("mlflow")

os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")

EXPERIMENT_NAME = "adult-pay-equity"
if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"{MLFLOW_DIR}/artifacts")
mlflow.set_experiment(EXPERIMENT_NAME)

print("MLflow data stored in:", MLFLOW_DIR)

# --- Evaluation and logging helpers ---

def classification_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def simple_params(estimator, prefix):
    # Keep only simple values (numbers, text, booleans, None) so MLflow can store them
    return {
        f"{prefix}__{key}": value
        for key, value in estimator.get_params(deep=False).items()
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def train_and_log(run_name, pipe, stage, remedy, log_model=False):
    # 1) Train
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # 2) Evaluate on validation (+ train ranking metrics to check overfitting)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_metrics = classification_metrics(y_val, val_pred, val_proba)

    train_proba = pipe.predict_proba(X_train)[:, 1]
    extra_metrics = {
        "train_roc_auc": roc_auc_score(y_train, train_proba),
        "train_pr_auc": average_precision_score(y_train, train_proba),
        "training_time_seconds": training_time,
    }

    # 3) Log to MLflow
    model = pipe.named_steps["model"]
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"stage": stage, "model_type": type(model).__name__, "remedy": remedy})
        mlflow.log_params(simple_params(model, "model"))
        if "sampler" in pipe.named_steps:
            mlflow.log_params(simple_params(pipe.named_steps["sampler"], "sampler"))
        mlflow.log_metrics({f"val_{name}": value for name, value in val_metrics.items()})
        mlflow.log_metrics(extra_metrics)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_val, val_pred, display_labels=["<=50K", ">50K"], ax=ax, colorbar=False, cmap=BLUES,
        )
        ax.set_title(run_name)
        mlflow.log_figure(fig, "confusion_matrix_val.png")
        plt.close(fig)

        # 4) Optionally save the whole fitted pipeline
        if log_model:
            # "cloudpickle" can save our custom AdultFeatureEngineer class.
            # Only reload model files you created yourself (pickle files can run code).
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

    return pd.Series({**val_metrics, **extra_metrics}, name=run_name)

# ==============================================================================
# Class imbalance
# ==============================================================================

def build_pipeline_with_sampler(model, sampler):
    # Same steps as build_pipeline, plus a resampling step applied during fit only
    steps = [("features", AdultFeatureEngineer()), ("preprocess", build_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)

remedies = {
    "none": dict(
        model=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        sampler=None,
    ),
    "class_weight": dict(
        model=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        sampler=None,
    ),
    "undersampling": dict(
        model=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        sampler=RandomUnderSampler(random_state=RANDOM_STATE),
    ),
    "smote": dict(
        model=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        sampler=SMOTE(random_state=RANDOM_STATE),
    ),
}

remedy_results = []
for remedy_name, parts in remedies.items():
    pipe = build_pipeline_with_sampler(parts["model"], parts["sampler"])
    result = train_and_log(
        run_name=f"logreg__{remedy_name}",
        pipe=pipe,
        stage="imbalance",
        remedy=remedy_name,
    )
    remedy_results.append(result.rename(remedy_name))

remedy_comparison = pd.DataFrame(remedy_results)
display(remedy_comparison[["precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy", "training_time_seconds"]].round(3))

# 4 metrics = 4 series colours; ROC-AUC stays in the table above
remedy_comparison[["precision", "recall", "f1", "pr_auc"]].plot(kind="bar", figsize=(10, 5))
plt.ylabel("Score (validation)")
plt.ylim(0, 1)
plt.title("Logistic Regression — effect of each imbalance remedy")
plt.xticks(rotation=0)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)  # below the chart
plt.show()

# ==============================================================================
# Three models
# ==============================================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
}

fitted_pipes = {}
model_results = []
for name, model in models.items():
    pipe = build_pipeline(model)
    result = train_and_log(
        run_name=name,
        pipe=pipe,
        stage="models",
        remedy="class_weight",
        log_model=True,
    )
    fitted_pipes[name] = pipe
    model_results.append(result)
    print(f"{name:20s} done in {result['training_time_seconds']:.1f}s")

comparison = pd.DataFrame(model_results).sort_values("pr_auc", ascending=False)

display(comparison[[
    "precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy", "training_time_seconds",
]].round(3))

comparison[["roc_auc", "pr_auc", "f1", "recall"]].plot(kind="bar", figsize=(10, 5))
plt.ylabel("Score (validation)")
plt.ylim(0, 1)
plt.title("Model comparison (class weights, default settings)")
plt.xticks(rotation=0)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)  # below the chart
plt.show()

# --- Precision–recall curves ---

fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in fitted_pipes.items():
    PrecisionRecallDisplay.from_estimator(pipe, X_val, y_val, name=name, ax=ax)

ax.axhline(y_val.mean(), linestyle="--", color=STEEL, label=f"Random model ({y_val.mean():.2f})")
ax.set_title("Precision–recall curves (validation)")
ax.legend(loc="lower left")
plt.show()

# --- Confusion matrices ---

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, pipe) in zip(axes, fitted_pipes.items()):
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_val, y_val, display_labels=["<=50K", ">50K"], ax=ax, colorbar=False, cmap=BLUES,
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()

# --- Overfitting check: train vs validation ---

overfit = comparison[["train_roc_auc", "roc_auc", "train_pr_auc", "pr_auc"]].rename(
    columns={"roc_auc": "val_roc_auc", "pr_auc": "val_pr_auc"}
)
overfit["roc_auc_gap"] = overfit["train_roc_auc"] - overfit["val_roc_auc"]
display(overfit.round(3))

# --- The runs in MLflow ---

runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

columns = [
    "tags.mlflow.runName", "tags.stage", "tags.remedy",
    "metrics.val_precision", "metrics.val_recall", "metrics.val_f1",
    "metrics.val_roc_auc", "metrics.val_pr_auc", "metrics.training_time_seconds",
]
display(runs[columns].sort_values("metrics.val_pr_auc", ascending=False).round(3))

lgbm_run_id = runs.loc[runs["tags.mlflow.runName"] == "LightGBM", "run_id"].iloc[0]
reloaded = mlflow.sklearn.load_model(f"runs:/{lgbm_run_id}/model")

print("Reloaded pipeline ROC-AUC on validation:",
      round(roc_auc_score(y_val, reloaded.predict_proba(X_val)[:, 1]), 3))
