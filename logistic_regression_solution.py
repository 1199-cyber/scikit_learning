import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv("dataset.csv")

#Task 1

pd.set_option('display.max_columns', None)
print(df.head())
print("\n")
print(df.info())
print("Number or rows :", len(df))
print("Number of columns:", len(df.columns))

#Task 2:

df = df.replace(["?", "invalid"], np.nan)
numeric_cols = [
    "study_hours",
    "sleep_hours",
    "attendance",
    "previous_score",
    "coffee_cups",
    "final_score",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.loc[(df["attendance"] > 100) | (df["attendance"] < 0), "attendance"] = np.nan
df = df.drop_duplicates()
columns_to_impute_mean = [
    "study_hours",
    "sleep_hours",
    "attendance",
    "previous_score",
]
for col in columns_to_impute_mean:
    df[col] = df[col].fillna(df[col].mean())
df["coffee_cups"] = df["coffee_cups"].fillna(df["coffee_cups"].median())
df = df.dropna(subset=["final_score"])
df["student_id"] = df["student_id"].astype(int)

#Task 3:

df["passed"] = (df["final_score"] >= 75).astype(int)
print("\n")
print(df[["student_id", "final_score", "passed"]].head())

#Task 4:

group_comparison = df.groupby("passed", observed=False)[["study_hours", "attendance"]].mean().reset_index()
print("\nAverage of students who did(1) and didn't pass(0):")
print(group_comparison.to_string(index=False))

#Task 5:

X = df[["study_hours", "sleep_hours", "attendance"]]
y = df["passed"]
model = LogisticRegression()
model.fit(X, y)
df["predictions"] = model.predict(X)
print("Predictions:")
print(df[["student_id", "passed", "predictions"]].head())

#Task 6:

accuracy = accuracy_score(y, df["predictions"])
print(f"\nAccuracy : {accuracy * 100:.2f}%")
print("\n")

#Task 7:

probabilities = model.predict_proba(X)[:, 1]
df["pass_probability"] = probabilities
print("Students and their calculated probability of passing:")
print(
    df[
        [
            "student_id",
            "study_hours",
            "attendance",
            "passed",
            "pass_probability",
        ]
    ].head(5)
)

#Task 10:

mean_attendance = df["attendance"].mean()
sleep_deprived_frequenters = df[
    (df["sleep_hours"] < 6) & (df["attendance"] > mean_attendance)
]
avg_probability = sleep_deprived_frequenters["pass_probability"].mean()
