"""Part 3: Hyperparameter tuning with Optuna and 5-fold stratified cross-validation.

Run from the repository root:  python -m src.part3_tuning
"""

from src.part2_imbalance_and_models import *  # noqa: F401,F403  (objects of the previous parts)

# ==============================================================================
# Why cross-validation for tuning
# ==============================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def cv_pr_auc(pipe):
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
    return scores.mean(), scores.std(), scores

default_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
    ),
}

rows = []
for name, model in default_models.items():
    mean, std, scores = cv_pr_auc(build_pipeline(model))
    rows.append({"model": name, "cv_pr_auc_mean": mean, "cv_pr_auc_std": std})
    print(f"{name:20s} fold scores: {np.round(scores, 3)}")

default_cv = pd.DataFrame(rows).set_index("model")
display(default_cv.round(4))

# ==============================================================================
# Optuna
# ==============================================================================

def make_logistic_regression(trial):
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 100, log=True),
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def make_random_forest(trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lightgbm(trial):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,  # needed for subsample to have an effect
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


MODEL_BUILDERS = {
    "Logistic Regression": make_logistic_regression,
    "Random Forest": make_random_forest,
    "LightGBM": make_lightgbm,
}

# Number of trials per model. Random Forest is the slowest, so it gets fewer.
N_TRIALS = {
    "Logistic Regression": 15,
    "Random Forest": 15,
    "LightGBM": 30,
}

# ==============================================================================
# Running the tuning
# ==============================================================================

def make_objective(model_name):
    builder = MODEL_BUILDERS[model_name]

    def objective(trial):
        pipe = build_pipeline(builder(trial))

        start = time.perf_counter()
        mean, std, scores = cv_pr_auc(pipe)
        duration = time.perf_counter() - start

        with mlflow.start_run(run_name=f"{model_name}: trial {trial.number}", nested=True):
            mlflow.set_tags({"stage": "tuning_trial", "model_type": model_name})
            mlflow.log_params(trial.params)
            mlflow.log_metrics({
                "cv_pr_auc_mean": mean,
                "cv_pr_auc_std": std,
                "cv_time_seconds": duration,
            })
            mlflow.log_dict({"fold_pr_auc": scores.tolist()}, "cv_fold_scores.json")

        print(f"  trial {trial.number:2d} | CV PR-AUC = {mean:.4f} ± {std:.4f} | {duration:5.1f}s")
        return mean

    return objective

studies = {}

for model_name in MODEL_BUILDERS:
    print(f"\n=== Tuning {model_name} ({N_TRIALS[model_name]} trials) ===")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=model_name,
    )

    with mlflow.start_run(run_name=f"tuning: {model_name}"):
        mlflow.set_tags({"stage": "tuning", "model_type": model_name})
        study.optimize(make_objective(model_name), n_trials=N_TRIALS[model_name])

        mlflow.log_params({f"best__{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_pr_auc", study.best_value)

        with plt.rc_context():   # keeps Optuna's "ggplot" style inside this plot only
            ax = optuna.visualization.matplotlib.plot_optimization_history(study)
            ax.set_title(f"Optimization history: {model_name}", loc="left")
            mlflow.log_figure(ax.figure, "optimization_history.png")
            plt.close(ax.figure)

    studies[model_name] = study
    print(f"Best CV PR-AUC for {model_name}: {study.best_value:.4f}")
    print("Best parameters:", study.best_params)

# --- The search, trial by trial ---

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, study) in zip(axes, studies.items()):
    trials = study.trials_dataframe()
    ax.plot(trials["number"], trials["value"], "o", alpha=0.6, label="trial")
    ax.plot(trials["number"], trials["value"].cummax(), label="best so far")
    ax.axhline(default_cv.loc[name, "cv_pr_auc_mean"], linestyle="--", color=STEEL,
               label="Part 2 default")
    ax.set_title(name)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV PR-AUC")
    ax.legend()

plt.tight_layout()
plt.show()

# --- Which hyperparameters mattered? ---

# Optuna's plots switch Matplotlib to its own "ggplot" style for the rest of the session:
# rc_context() keeps that change inside this cell, so the following figures keep our theme.
with plt.rc_context():
    for name in ["Random Forest", "LightGBM"]:
        ax = optuna.visualization.matplotlib.plot_param_importances(studies[name])
        ax.set_title(f"Hyperparameter importance: {name}", loc="left")  # replaces Optuna's own title
        ax.set_facecolor("white")                 # Optuna draws in its own grey style: apply ours
        ax.grid(False)
        for bar in ax.patches:
            bar.set_color(NAVY)
        if ax.get_legend() is not None:           # Optuna's legend adds nothing here
            ax.get_legend().remove()
        plt.show()

# All LightGBM trials, best first
display(studies["LightGBM"].trials_dataframe().sort_values("value", ascending=False).head(10).round(4))

# ==============================================================================
# Tuned vs default models on the validation set
# ==============================================================================

def fit_and_evaluate(name, pipe, stage, params=None, log_model=False):
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    val_proba = pipe.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_metrics = classification_metrics(y_val, val_pred, val_proba)

    train_proba = pipe.predict_proba(X_train)[:, 1]
    extra_metrics = {
        "train_pr_auc": average_precision_score(y_train, train_proba),
        "training_time_seconds": training_time,
    }

    with mlflow.start_run(run_name=f"{name} ({stage})") as run:
        mlflow.set_tags({"stage": stage, "model_type": name, "remedy": "class_weight"})
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({f"val_{metric}": value for metric, value in val_metrics.items()})
        mlflow.log_metrics(extra_metrics)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_val, val_pred, display_labels=["<=50K", ">50K"], ax=ax, colorbar=False, cmap=BLUES,
        )
        ax.set_title(f"{name} ({stage})")
        mlflow.log_figure(fig, "confusion_matrix_val.png")
        plt.close(fig)
        if log_model:
            # cloudpickle can save our custom AdultFeatureEngineer class (see Part 2)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

    return {**val_metrics, **extra_metrics, "run_id": run.info.run_id}

rows = []
tuned_pipes = {}   # kept in memory for Part 4
for name, study in studies.items():
    best_model = MODEL_BUILDERS[name](optuna.trial.FixedTrial(study.best_params))
    tuned_pipe = build_pipeline(best_model)
    tuned = fit_and_evaluate(name, tuned_pipe, stage="tuned", params=study.best_params, log_model=True)
    tuned_pipes[name] = tuned_pipe
    default = fit_and_evaluate(name, build_pipeline(default_models[name]), stage="default")

    for version, metrics in [("default", default), ("tuned", tuned)]:
        rows.append({"model": name, "version": version, **metrics})

results = pd.DataFrame(rows).set_index(["model", "version"])
display(results[["precision", "recall", "f1", "roc_auc", "pr_auc", "train_pr_auc", "training_time_seconds"]].round(3))

summary = pd.DataFrame({
    "default_cv_pr_auc": default_cv["cv_pr_auc_mean"],
    "tuned_cv_pr_auc": pd.Series({name: s.best_value for name, s in studies.items()}),
    "default_val_pr_auc": results.xs("default", level="version")["pr_auc"],
    "tuned_val_pr_auc": results.xs("tuned", level="version")["pr_auc"],
})
summary["gain_on_val"] = summary["tuned_val_pr_auc"] - summary["default_val_pr_auc"]
display(summary.round(4))

summary[["default_val_pr_auc", "tuned_val_pr_auc"]].plot(kind="bar", figsize=(9, 5))
plt.ylabel("Validation PR-AUC")
plt.ylim(0.7, 0.9)
plt.title("Effect of tuning (validation set)")
plt.xticks(rotation=0)
plt.show()

# ==============================================================================
# Final choice
# ==============================================================================

tuned_results = results.xs("tuned", level="version")
FINAL_MODEL_NAME = tuned_results["pr_auc"].idxmax()
final_pipe = tuned_pipes[FINAL_MODEL_NAME]    # the tuned pipeline, already fitted on the training set
final_run_id = tuned_results.loc[FINAL_MODEL_NAME, "run_id"]

print("Final model:", FINAL_MODEL_NAME)
print("Validation PR-AUC:", round(tuned_results.loc[FINAL_MODEL_NAME, "pr_auc"], 4))
print("Best parameters:", studies[FINAL_MODEL_NAME].best_params)

# Mark the chosen run in MLflow so Part 4 can find it
with mlflow.start_run(run_id=final_run_id):
    mlflow.set_tag("final_model", "true")

runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
print("Total runs in the experiment:", len(runs))
display(runs["tags.stage"].value_counts())

display(runs[runs["tags.stage"] == "tuned"][[
    "tags.mlflow.runName", "tags.final_model", "metrics.val_pr_auc", "metrics.val_roc_auc", "metrics.val_f1",
]].round(4))

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
