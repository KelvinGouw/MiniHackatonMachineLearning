import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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


def main():
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

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    best_name = None
    best_pipeline = None
    best_acc = -1

    for name, model in candidates.items():
        pipeline = build_pipeline(categorical_cols, numeric_cols, model)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_pipeline = pipeline
            best_name = name

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": best_pipeline,
            "model_name": best_name,
            "features": feature_cols,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "accuracy": best_acc,
        },
        MODEL_PATH,
    )

    print(f"Saved model to {MODEL_PATH}")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
