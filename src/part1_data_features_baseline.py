"""Part 1: Data exploration, column choice, feature engineering, pipeline and baseline.

Run from the repository root:  python -m src.part1_data_features_baseline
"""

from src.common import *  # noqa: F401,F403  (objects of the previous parts)

# ==============================================================================
# Dataset presentation
# ==============================================================================
# --- Sensitive attributes ---

# Download the dataset from OpenML
adult = fetch_openml(name="adult", version=2, as_frame=True)

X_raw = adult.data.copy()
y_raw = adult.target.copy()

print("Feature matrix shape:", X_raw.shape)
print("Target values:", y_raw.unique().tolist())

# --- Encode the target ---

X = X_raw.copy()
y = (y_raw.astype(str).str.strip() == ">50K").astype(int).rename(TARGET)

df = pd.concat([X, y], axis=1)

display(df.head())

# ==============================================================================
# Exploratory Data Analysis
# ==============================================================================
# --- First inspection ---

def basic_summary(df):
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "target_rate": float(df[TARGET].mean()),
    }

display(basic_summary(df))

# --- Column types and descriptive statistics ---

numeric_cols = X.select_dtypes("number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

print("Numeric columns    :", numeric_cols)
print("Categorical columns:", categorical_cols)

print("\nData types:")
display(df.dtypes)

print("\nDescriptive statistics (numeric):")
display(df[numeric_cols].describe().T)

print("\nNumber of categories per categorical column:")
display(X[categorical_cols].nunique().sort_values(ascending=False))

# --- Explore the target ---

def plot_target_distribution(y):
    counts = y.value_counts().sort_index()
    print(f"Share of people earning >50K: {y.mean():.2%}")

    counts.plot(kind="bar")
    plt.xticks([0, 1], ["<=50K (0)", ">50K (1)"], rotation=0)
    plt.ylabel("Number of people")
    plt.title("Target distribution")
    plt.show()

plot_target_distribution(y)

# --- Missing values ---

def missing_summary(df):
    missing = df.isna().sum()
    table = pd.DataFrame({
        "n_missing": missing,
        "pct_missing": (missing / len(df) * 100).round(2),
    })
    return table[table["n_missing"] > 0].sort_values("n_missing", ascending=False)

display(missing_summary(df))

# Are workclass and occupation missing for the same people?
display(pd.crosstab(
    df["workclass"].isna(),
    df["occupation"].isna(),
    rownames=["workclass missing"],
    colnames=["occupation missing"],
))

# Does missingness tell us something about income?
for col in ["workclass", "occupation", "native-country"]:
    rate_missing = df.loc[df[col].isna(), TARGET].mean()
    rate_known = df.loc[df[col].notna(), TARGET].mean()
    print(f"{col:15s} | >50K rate when missing: {rate_missing:.2%} | when known: {rate_known:.2%}")

# --- Explore numeric predictors ---

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for ax, col in zip(axes.ravel(), numeric_cols):
    ax.hist(df[col], bins=40)
    ax.set_title(col)
    ax.set_ylabel("Number of people")

plt.suptitle("Distributions of the numeric features")
plt.tight_layout()
plt.show()

for col in ["capital-gain", "capital-loss"]:
    print(f"{col}: {(df[col] == 0).mean():.1%} of people have exactly 0, max = {df[col].max():,}")

# --- Which features are related to income? ---

def rate_by_group(df, feature):
    return (
        df.groupby(feature, observed=True)[TARGET]
        .agg(n_people="size", rate_over_50k="mean")
        .sort_values("rate_over_50k", ascending=False)
    )

for feature in ["marital-status", "relationship", "education-num"]:
    print(f"\n=== {feature} ===")
    display(rate_by_group(df, feature).round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

rate_by_group(df, "marital-status")["rate_over_50k"].sort_values().plot.barh(ax=axes[0])
axes[0].set_title(">50K rate by marital status")
axes[0].set_xlabel("Share earning >50K")

rate_by_group(df, "education-num")["rate_over_50k"].sort_index().plot.bar(ax=axes[1])
axes[1].set_title(">50K rate by education level")
axes[1].set_xlabel("education-num")
axes[1].set_ylabel("Share earning >50K")

plt.tight_layout()
plt.show()

# --- Sensitive attributes: sex and race ---

for feature in ["sex", "race"]:
    print(f"\n=== {feature} ===")
    display(rate_by_group(df, feature).round(3))

# --- Does the gap remain between comparable people? ---

rate_by_education_and_sex = (
    df.groupby(["education-num", "sex"], observed=True)[TARGET]
    .mean()
    .unstack()
)

rate_by_education_and_sex.plot(marker="o", figsize=(8, 5))
plt.xlabel("education-num")
plt.ylabel("Share earning >50K")
plt.title(">50K rate by education level, men vs women")
plt.show()

display(rate_by_education_and_sex.loc[[9, 10, 13, 14]].round(3))

# Same comparison inside the most common occupations
top_occupations = df["occupation"].value_counts().head(6).index
display((
    df[df["occupation"].isin(top_occupations)]
    .groupby(["occupation", "sex"], observed=True)[TARGET]
    .mean()
    .unstack()
    .round(3)
))

# --- Which columns may the reference model use? ---
# --- Is relationship a proxy for sex? ---

display(pd.crosstab(df["relationship"], df["sex"]))

# --- Redundant and meaningless columns ---

# Each education label should map to exactly one education-num value
education_map = df.groupby("education", observed=True)["education-num"].nunique()
print("Max number of education-num values per education label:", education_map.max())

display(df.groupby("education", observed=True)["education-num"].first().sort_values())

# fnlwgt: >50K rate across 10 equal-sized groups (deciles) of fnlwgt
fnlwgt_deciles = pd.qcut(df["fnlwgt"], q=10)
display(df.groupby(fnlwgt_deciles, observed=True)[TARGET].mean().round(3))

# --- Decision: the columns of the reference pay model ---

# --- Leakage audit: could we know each column at prediction time? ---

# ==============================================================================
# Train / Validation / Test Split
# ==============================================================================
# --- Create the split ---

def make_train_val_test_split(X, y, test_size=0.20, val_size=0.20, random_state=42):
    # 1) Cut off the test set first
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    # 2) Split the rest into train and validation.
    #    val_size is a share of the WHOLE dataset, so we rescale it.
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_rest,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
    X, y,
    test_size=0.20,
    val_size=0.20,
    random_state=RANDOM_STATE,
)

for name, X_part, y_part in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    print(f"{name:10s} shape: {X_part.shape} | >50K rate: {y_part.mean():.3f}")

# ==============================================================================
# Feature Engineering
# ==============================================================================
# --- The AdultFeatureEngineer transformer ---

# Columns the reference pay model is NOT allowed to use
PROTECTED_ATTRIBUTES = ("sex", "race", "native-country")     # kept aside for the audit only
PROXY_COLUMNS = ("relationship", "marital-status")            # encode sex / not job-related
NOT_USEFUL_COLUMNS = ("fnlwgt", "education")                  # sampling weight / duplicate
REFERENCE_DROP = PROTECTED_ATTRIBUTES + PROXY_COLUMNS + NOT_USEFUL_COLUMNS


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns=REFERENCE_DROP):
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        X = X.copy()  # never modify the original DataFrame

        # 1) Drop protected, proxy and useless columns
        X = X.drop(columns=[c for c in self.drop_columns if c in X.columns])

        # 2) New features
        gain = X["capital-gain"].fillna(0)
        loss = X["capital-loss"].fillna(0)
        X["capital_net"] = gain - loss
        X["has_capital"] = ((gain > 0) | (loss > 0)).astype(int)
        X["log_capital_gain"] = np.log1p(gain)
        X["works_overtime"] = (X["hours-per-week"] > 40).astype(int)
        X["is_part_time"] = (X["hours-per-week"] < 35).astype(int)

        # 3) Store categorical columns as plain text (object) with NaN for missing,
        #    so the imputer and one-hot encoder handle them consistently
        for col in X.select_dtypes(exclude="number").columns:
            X[col] = X[col].astype("object").where(X[col].notna(), np.nan)

        return X

engineer = AdultFeatureEngineer()
X_train_fe = engineer.fit_transform(X_train)

print("Before:", X_train.shape, "| After:", X_train_fe.shape)
print("Columns used by the model:", list(X_train_fe.columns))
new_features = ["capital_net", "has_capital", "log_capital_gain", "works_overtime", "is_part_time"]
display(X_train_fe[new_features].head())

# --- Do the new features carry signal? ---

train_fe = X_train_fe.assign(**{TARGET: y_train.values})

for feature in ["has_capital", "works_overtime", "is_part_time"]:
    print(f"\n=== {feature} ===")
    display(rate_by_group(train_fe, feature).round(3))

# Do men and women differ on these legitimate factors? (training set)
display(X_train_fe.assign(sex=X_train["sex"].values).groupby("sex", observed=True)[
    ["works_overtime", "is_part_time", "has_capital", "hours-per-week", "education-num", "age"]
].mean().round(3))

# ==============================================================================
# Preprocessing Pipeline
# ==============================================================================
# --- Build the pipeline ---

def build_preprocessor():
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",  # unseen category later -> no crash
            min_frequency=0.005,                   # categories under 0.5% are grouped together
            sparse_output=False,
        )),
    ])

    # Columns are selected by type AFTER feature engineering,
    # so the new features are included automatically.
    return ColumnTransformer([
        ("num", numeric_steps, make_column_selector(dtype_include="number")),
        ("cat", categorical_steps, make_column_selector(dtype_exclude="number")),
    ])


def build_pipeline(model):
    return Pipeline([
        ("features", AdultFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])

baseline_pipe = build_pipeline(
    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
)
display(baseline_pipe)

# --- Sanity check ---

preprocessing = baseline_pipe[:-1]
X_train_ready = preprocessing.fit_transform(X_train, y_train)

feature_names = baseline_pipe.named_steps["preprocess"].get_feature_names_out()
print("Shape after preprocessing:", X_train_ready.shape)
print("First feature names:", list(feature_names[:8]))

numeric_part = X_train_ready[:, [name.startswith("num__") for name in feature_names]]
print("\nLargest absolute training mean (numeric):", np.abs(numeric_part.mean(axis=0)).max().round(6))
print("Training std range (numeric):", numeric_part.std(axis=0).min().round(3), "to", numeric_part.std(axis=0).max().round(3))

# ==============================================================================
# Baseline Models
# ==============================================================================
# --- Evaluation function ---

def classification_metrics(y_true, y_pred, y_proba):
    return pd.Series({
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    })


def evaluate_pipeline(name, pipe, X_train, y_train, X_eval, y_eval):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_eval)
    y_proba = pipe.predict_proba(X_eval)[:, 1]
    return classification_metrics(y_eval, y_pred, y_proba).rename(name)

# --- Train and compare the baselines ---

dummy_pipe = build_pipeline(DummyClassifier(strategy="most_frequent"))

dummy_results = evaluate_pipeline("Dummy (majority)", dummy_pipe, X_train, y_train, X_val, y_val)
logreg_results = evaluate_pipeline("Logistic Regression", baseline_pipe, X_train, y_train, X_val, y_val)

comparison = pd.concat([dummy_results, logreg_results], axis=1).T
display(comparison.round(3))

# --- Train vs validation: is the baseline overfitting? ---

train_results = classification_metrics(
    y_train,
    baseline_pipe.predict(X_train),
    baseline_pipe.predict_proba(X_train)[:, 1],
).rename("train")

display(pd.concat([train_results, logreg_results.rename("validation")], axis=1).round(3))

# --- Confusion matrix and ROC curve ---

y_val_pred = baseline_pipe.predict(X_val)
y_val_proba = baseline_pipe.predict_proba(X_val)[:, 1]

cm = confusion_matrix(y_val, y_val_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True negatives : {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives : {tp}")

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"]).plot(cmap=BLUES, colorbar=False)
plt.title("Logistic Regression: Confusion matrix (validation)")
plt.show()

RocCurveDisplay.from_predictions(y_val, y_val_proba)
plt.title("Logistic Regression: ROC curve (validation)")
plt.show()

# --- Takeaways of Part 1 ---
