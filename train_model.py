
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

n = 2000

data = pd.DataFrame({
    "attendance": np.random.uniform(40, 100, n),
    "internal_marks": np.random.uniform(20, 100, n),
    "assignment_marks": np.random.uniform(20, 100, n),
    "previous_score": np.random.uniform(20, 100, n)
})

score = (
    data["attendance"] * 0.20 +
    data["internal_marks"] * 0.35 +
    data["assignment_marks"] * 0.15 +
    data["previous_score"] * 0.30
)

data["result"] = (score >= 50).astype(int)

X = data.drop(columns=["result"])
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "student_result_model.pkl")
data.to_csv("student_results.csv", index=False)

with open("metrics.json", "w") as f:
    json.dump({"accuracy": float(accuracy)}, f, indent=4)

print("Model training completed.")
print("Accuracy:", accuracy)
