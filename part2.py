import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference
import matplotlib.pyplot as plt

# Reused from part 1
df = load_and_clean("data")
df = prepare(df)

# Features/target
target = "y"
sensitive = df["sex"]
X = df.drop(columns=["income", target, "sex", "sex_male"], errors="ignore")
y = df[target]

# Split
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42, stratify=y
)

# Column types
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocess + model
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
)

clf = Pipeline([
    ("prep", preprocess),
    ("lr", LogisticRegression(max_iter=1000)),
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Required metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred, zero_division=0):.3f}")

# Fairness metrics
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
eo_diff = equal_opportunity_difference(y_test, y_pred, sensitive_features=s_test)
print(f"Demographic Parity Difference: {dp_diff:.3f}")
print(f"Equal Opportunity Difference: {eo_diff:.3f}")

# Calibration (overall)
print(f"Calibration (Brier score): {brier_score_loss(y_test, y_prob):.3f}")

# Visualization: confusion matrices by subgroup
groups = sorted(s_test.unique())
fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
if len(groups) == 1:
    axes = [axes]

for ax, g in zip(axes, groups):
    mask = (s_test == g)
    cm = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {g}")

plt.tight_layout()
plt.show()
