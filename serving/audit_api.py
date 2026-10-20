"""Pay-equity audit API.

Run locally with:   uvicorn audit_api:app --reload   (from inside the serving/ folder)
Then open http://127.0.0.1:8000 for the web interface,
or http://127.0.0.1:8000/docs for the interactive documentation of the endpoints.
"""
import json
from pathlib import Path
from typing import List, Optional

import cloudpickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).parent
MIN_GROUP_SIZE = 100         # smaller groups are reported but marked as too small (same rule as part 4)

with open(HERE / "audit_model.pkl", "rb") as f:
    MODEL = cloudpickle.load(f)
with open(HERE / "model_info.json") as f:
    INFO = json.load(f)
THRESHOLD = INFO["threshold"]

# real people of the test set (never seen during training), used by the web interface
PEOPLE = pd.read_csv(HERE / "test_people.csv", index_col="id")

EDUCATION_LEVELS = {
    1: "Preschool", 2: "1st-4th", 3: "5th-6th", 4: "7th-8th", 5: "9th", 6: "10th", 7: "11th", 8: "12th",
    9: "HS-grad", 10: "Some-college", 11: "Assoc-voc", 12: "Assoc-acdm", 13: "Bachelors", 14: "Masters",
    15: "Prof-school", 16: "Doctorate",
}


# ---------- input format ----------
class Employee(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # job-related profile: the only information used by the model
    age: int = Field(ge=16, le=100)
    workclass: Optional[str] = None
    education_num: int = Field(alias="education-num", ge=1, le=16)
    occupation: Optional[str] = None
    capital_gain: float = Field(0, alias="capital-gain", ge=0)
    capital_loss: float = Field(0, alias="capital-loss", ge=0)
    hours_per_week: float = Field(alias="hours-per-week", gt=0, le=99)

    # used only to group the audit results, never by the model
    sex: Optional[str] = None
    race: Optional[str] = None

    # actual pay level (needed for /audit): 1 if the employee earns >50K
    income_over_50k: Optional[int] = Field(None, ge=0, le=1)


class PredictRequest(BaseModel):
    employees: List[Employee]


class AuditRequest(BaseModel):
    employees: List[Employee]
    group_by: List[str] = ["sex", "race"]


# ---------- helpers ----------
def to_dataframe(employees: List[Employee]) -> pd.DataFrame:
    return pd.DataFrame([e.model_dump(by_alias=True) for e in employees])


def expected_probabilities(data: pd.DataFrame) -> np.ndarray:
    return MODEL.predict_proba(data)[:, 1]


def bootstrap_gap_ci(actual, expected, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(actual)
    gaps = [actual[idx].mean() - expected[idx].mean()
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.percentile(gaps, [2.5, 97.5])


# ---------- the API ----------
app = FastAPI(title="Pay-equity audit API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": INFO["model_type"], "threshold": THRESHOLD}


@app.post("/predict")
def predict(request: PredictRequest):
    data = to_dataframe(request.employees)
    proba = expected_probabilities(data)
    return {
        "predictions": [
            {"expected_probability": round(float(p), 4), "expected_over_50k": bool(p >= THRESHOLD)}
            for p in proba
        ]
    }


@app.post("/audit")
def audit(request: AuditRequest):
    data = to_dataframe(request.employees)
    if data["income_over_50k"].isna().any():
        raise HTTPException(status_code=422, detail="Every employee needs income_over_50k (0 or 1) for an audit.")

    actual = data["income_over_50k"].to_numpy(dtype=int)
    expected = expected_probabilities(data)
    flagged = (expected >= THRESHOLD) & (actual == 0)       # expected >50K but paid <=50K

    results = {}
    for column in request.group_by:
        if column not in data or data[column].isna().all():
            continue
        groups = []
        for group, idx in data.groupby(column).indices.items():
            low, high = bootstrap_gap_ci(actual[idx], expected[idx])
            groups.append({
                "group": group,
                "n": int(len(idx)),
                "actual_rate": round(float(actual[idx].mean()), 4),
                "expected_rate": round(float(expected[idx].mean()), 4),
                "unexplained_gap_pts": round(float(100 * (actual[idx].mean() - expected[idx].mean())), 2),
                "ci_95_pts": [round(float(100 * low), 2), round(float(100 * high), 2)],
                "significant": bool(high < 0 or low > 0),
                "enough_data": bool(len(idx) >= MIN_GROUP_SIZE),
                "n_flagged_underpaid": int(flagged[idx].sum()),
            })
        results[column] = sorted(groups, key=lambda g: g["unexplained_gap_pts"])

    return {
        "n_employees": int(len(data)),
        "threshold": THRESHOLD,
        "groups": results,
        "flagged_underpaid_rows": np.flatnonzero(flagged).tolist(),
        "expected_probabilities": [round(float(p), 4) for p in expected],
    }


# ---------- web interface ----------
@app.get("/", include_in_schema=False)
def web_interface():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/options")
def options():
    """Choices offered by the web form, taken from the data (most frequent first)."""
    def choices(column):
        return PEOPLE[column].value_counts().index.tolist()

    return {
        "workclass": choices("workclass"),
        "occupation": choices("occupation"),
        "sex": choices("sex"),
        "race": choices("race"),
        "education_levels": EDUCATION_LEVELS,
        "n_people": int(len(PEOPLE)),
        "model": INFO,
        "min_group_size": MIN_GROUP_SIZE,
    }


@app.get("/people")
def people(
    n: int = Query(1, ge=1, le=len(PEOPLE)),
    seed: Optional[int] = None,
    sex: Optional[str] = None,
    race: Optional[str] = None,
    workclass: Optional[str] = None,
    occupation: Optional[str] = None,
):
    """Random real people of the test set, with their actual pay level. Optional filters on one value per column."""
    pool = PEOPLE
    for column, value in [("sex", sex), ("race", race), ("workclass", workclass), ("occupation", occupation)]:
        if value:
            pool = pool[pool[column] == value]
    if pool.empty:
        raise HTTPException(status_code=404, detail="Nobody in the test set matches these filters.")

    sample = pool.sample(min(n, len(pool)), random_state=seed).reset_index()
    return {
        "n_matching": int(len(pool)),
        "people": json.loads(sample.to_json(orient="records")),   # missing values become null
    }
