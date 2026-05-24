import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.path.join("dataset", "student_mental_health_burnout.csv")
MODEL_PATH = os.path.join("models", "burnout_model.joblib")
SELECTED_FEATURES = [
    "daily_study_hours",
    "daily_sleep_hours",
    "screen_time_hours",
    "physical_activity_hours",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "academic_pressure_score",
    "financial_stress_score",
    "social_support_score",
    "sleep_quality",
    "attendance_percentage",
]


def build_pipeline(categorical_cols, numeric_cols, model):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_save():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()

    target_col = "burnout_level"
    df = df.dropna(subset=[target_col])

    feature_cols = SELECTED_FEATURES
    categorical_cols = ["stress_level", "sleep_quality"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    best_name = "logistic_regression"
    best_pipeline = build_pipeline(categorical_cols, numeric_cols, model)
    best_pipeline.fit(X_train, y_train)
    preds = best_pipeline.predict(X_test)
    best_acc = accuracy_score(y_test, preds)

    bundle = {
        "model": best_pipeline,
        "model_name": best_name,
        "features": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "accuracy": best_acc,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=("gzip", 3))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {best_acc:.4f}")
    return bundle


def main():
    train_and_save()


if __name__ == "__main__":
    main()
