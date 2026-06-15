import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
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

corr_study = np.corrcoef(df["study_hours"], df["final_score"])[0, 1]
corr_sleep = np.corrcoef(df["sleep_hours"], df["final_score"])[0, 1]
print(f"\nCorrelation (Study Hours vs Final Score): {corr_study:.4f}")
print(f"Correlation (Sleep Hours vs Final Score): {corr_sleep:.4f}")
print("\n")

#Task 4:

attendance_threshold = 85.0
df["attendance_group"] = np.where(
    df["attendance"] < attendance_threshold, "Low Attendance", "High Attendance"
)
attendance_impact = (
    df.groupby("attendance_group", observed=False)["final_score"].mean().reset_index()
)
print("Average final score by attendance group:")
print(attendance_impact.to_string(index=False))

#Task 5:

coffee_impact = (
    df.groupby("coffee_cups", observed=False)["final_score"].mean().reset_index()
)
print("\nAverage final score based on coffee cups consumed:")
print(coffee_impact.to_string(index=False))

#Task 6:

numeric_df = df[
    [
        "study_hours",
        "sleep_hours",
        "attendance",
        "previous_score",
        "coffee_cups",
        "final_score",
    ]
]
correlation_matrix = numeric_df.corr()
print("\nFull correlation matrix:")
print(correlation_matrix["final_score"].to_string())

#Task 7:

X = df[["study_hours", "sleep_hours", "attendance"]]
y = df["final_score"]
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)
print(f"\nMean Absolute Error (MAE): {mae:.2f} points")
print(f"R-squared Score (R²): {r2:.4f}")

#Task 8:

feature_names = ["study_hours", "sleep_hours", "attendance"]
coefficients = model.coef_
intercept = model.intercept_
weights_summary = pd.DataFrame(
    {"Feature": feature_names, "Coefficient (Weight)": coefficients}
)
print("\nModel attributes:")
print(weights_summary.to_string(index=False))
print(f"Baseline intercept value: {intercept:.2f}")

#Task 9

mean_previous = df["previous_score"].mean()
filtered_students = df[
    (df["previous_score"] > mean_previous) & (df["study_hours"] < 6)
]
target_avg_score = filtered_students["final_score"].mean()


