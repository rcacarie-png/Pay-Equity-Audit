"""Part 5: Export of the calibrated model and tests of the audit API (serving/audit_api.py).

Run from the repository root:  python -m src.part5_serving
"""

from src.part4_explain_calibrate_audit import *  # noqa: F401,F403  (objects of the previous parts)

# Script only: store the code of our custom transformers inside the pickle (as the notebook does),
# so that serving/ can load the model without the src/ package.
for _module in {AdultFeatureEngineer.__module__}:
    if _module != "__main__":
        cloudpickle.register_pickle_by_value(sys.modules[_module])

# ==============================================================================
# Export the model
# ==============================================================================

SERVING_DIR = "serving"
os.makedirs(SERVING_DIR, exist_ok=True)

with open(f"{SERVING_DIR}/audit_model.pkl", "wb") as f:
    cloudpickle.dump(calibrated_model, f)

model_info = {
    "model_type": FINAL_MODEL_NAME,
    "calibration_method": CALIBRATION_METHOD,
    "threshold": THRESHOLD,
    "excluded_columns": list(REFERENCE_DROP),
    "test_roc_auc": float(final_results.loc["roc_auc", "test"]),
}
with open(f"{SERVING_DIR}/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

display(model_info)

# Real people of the test set for the web interface (serving/static/index.html): the people it draws
# for the audit and the "Load a real person" button. Same columns as the API input.
people_columns = ["age", "workclass", "education-num", "occupation", "capital-gain", "capital-loss",
                  "hours-per-week", "sex", "race"]
test_people = X_test[people_columns].assign(**{TARGET: y_test.values})
test_people.index.name = "id"
test_people.to_csv(f"{SERVING_DIR}/test_people.csv")

# ==============================================================================
# Write the API
# ==============================================================================

# The API itself is in serving/audit_api.py (written by %%writefile in the notebook).

# ==============================================================================
# Test the API
# ==============================================================================

from fastapi.testclient import TestClient

sys.path.insert(0, SERVING_DIR)
import audit_api

client = TestClient(audit_api.app)
display(client.get("/health").json())

profile = {
    "age": 42, "workclass": "Private", "education-num": 13, "occupation": "Exec-managerial",
    "capital-gain": 0, "capital-loss": 0, "hours-per-week": 45,
}
response = client.post("/predict", json={"employees": [
    {**profile, "sex": "Male", "race": "White"},
    {**profile, "sex": "Female", "race": "Black"},
]})
display(response.json())

bad = client.post("/predict", json={"employees": [{**profile, "age": 7}]})
print("Status code:", bad.status_code)
display(bad.json()["detail"][0]["msg"])

columns = ["age", "workclass", "education-num", "occupation", "capital-gain", "capital-loss",
           "hours-per-week", "sex", "race"]

company = X_test.sample(2000, random_state=RANDOM_STATE)[columns].copy()
company["income_over_50k"] = y_test.loc[company.index].values

# to_json turns missing values into null and numbers into plain JSON numbers
company_records = json.loads(company.to_json(orient="records"))
print(json.dumps(company_records[0], indent=2))

audit_response = client.post("/audit", json={"employees": company_records}).json()
print("Employees audited:", audit_response["n_employees"])
print("Flagged as potentially underpaid:", len(audit_response["flagged_underpaid_rows"]))

for attribute, groups in audit_response["groups"].items():
    print(f"\n=== {attribute} ===")
    display(pd.DataFrame(groups).set_index("group"))

# ==============================================================================
# Run the API as a real server (optional, on your computer)
# ==============================================================================

# Download the serving folder from Colab
try:
    from google.colab import files
    archive = shutil.make_archive("/content/serving", "zip", SERVING_DIR)
    files.download(archive)
except ImportError:
    print("Not in Colab: the API files are in", os.path.abspath(SERVING_DIR))

# ==============================================================================
# Make the MLflow history portable (last step)
# ==============================================================================

# MLflow saves the full path of the artifacts folder of this computer (for example /Users/rob/...).
# We replace it by the relative path "mlflow/artifacts", so the mlflow/ folder committed in the repository
# opens on any computer with: mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db  (run from the repository root)
import sqlite3

repo_root = os.path.abspath(".") + os.sep
connection = sqlite3.connect(os.path.join(MLFLOW_DIR, "mlflow.db"))
connection.execute("UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, '')", (repo_root,))
connection.execute("UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, '')", (repo_root,))
connection.execute("UPDATE logged_models SET artifact_location = REPLACE(artifact_location, ?, '')", (repo_root,))
connection.commit()
connection.close()

# the saved models also write the full path in their description file (MLmodel), we clean it the same way
for folder, subfolders, files in os.walk(os.path.join(MLFLOW_DIR, "artifacts")):
    if "MLmodel" in files:
        path = os.path.join(folder, "MLmodel")
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(text.replace(repo_root, ""))
print("MLflow paths made relative, the folder mlflow/ can be committed")
