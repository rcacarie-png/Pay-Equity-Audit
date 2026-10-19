"""Shared setup: imports, constants and the graphic identity of the agency (blues and greys).

Every part script starts with `from src.<previous part> import *`, so running a part runs
everything it depends on, in order, in the same Python process (like "Run all" in the notebook).
"""
import os
import sys
import json
import time
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import optuna
import shap
import cloudpickle

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline   # a Pipeline that accepts resampling steps
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)

RANDOM_STATE = 42
TARGET = "income_over_50k"

pd.set_option("display.max_columns", 50)
set_config(display="diagram")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- Graphic identity: blues and greys ----
import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

NAVY, SKY, TEAL, VIOLET = "#1C5AA3", "#74B9EE", "#00927F", "#8170D8"   # series colours, in this order
STEEL, SILVER, INK = "#5F6B7A", "#C5CDD6", "#2B2F36"                   # greys: reference lines, axes, text
SERIES_COLORS = [NAVY, SKY, TEAL, VIOLET]

BLUES = LinearSegmentedColormap.from_list("agency_blues", ["#EEF3F9", NAVY])   # light -> dark, for magnitudes
mpl.colormaps.register(BLUES, force=True)

plt.rcParams.update({
    "axes.prop_cycle": cycler(color=SERIES_COLORS),
    "image.cmap": "agency_blues",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": SILVER,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": STEEL,
    "ytick.color": STEEL,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "patch.edgecolor": "white",
})

# SHAP plots: navy = pushes towards >50K, violet = pushes towards <=50K
try:
    from shap.plots._style import set_style
    set_style(
        primary_color_positive=NAVY, primary_color_negative=VIOLET,
        secondary_color_positive="#8FB0D9", secondary_color_negative="#C3B9EE",
        hlines_color="#E3E8EE", vlines_color=SILVER, tick_labels_color=STEEL,
    )
except ImportError:
    print("Older shap version: SHAP plots keep their default colours.")


# ==============================================================================
# Script helpers (not in the notebook): show tables and save figures
# ==============================================================================
import inspect

matplotlib_backend = os.environ.get("MPLBACKEND")
if matplotlib_backend is None:
    plt.switch_backend("Agg")          # no window: figures are saved to reports/figures/

pd.set_option("display.width", 160)
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
_figure_counter = {}


def display(obj):
    """Script version of the notebook's display(): print tables and values."""
    if obj is not None:
        print(obj)
        print()


def _save_open_figures(*args, **kwargs):
    """Script version of plt.show(): save every open figure as reports/figures/<part>_<nn>.png."""
    caller = inspect.stack()[1].frame.f_globals.get("__file__", "figure")
    part = os.path.splitext(os.path.basename(caller))[0].split("_")[0]
    for number in plt.get_fignums():
        _figure_counter[part] = _figure_counter.get(part, 0) + 1
        path = os.path.join(FIGURES_DIR, f"{part}_{_figure_counter[part]:02d}.png")
        plt.figure(number).savefig(path, dpi=150, bbox_inches="tight")
        print("Figure saved:", path)
    plt.close("all")


plt.show = _save_open_figures
