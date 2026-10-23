"""Task 1 - Model Training.

Loads Titanic train.csv, preprocesses, trains a RandomForest classifier,
evaluates accuracy on a held-out test split, and saves the COMPLETE pipeline
(preprocessing + model) to model.pkl using Joblib.

The held-out test split is also saved to data/unseen_data.csv so that the
inference tasks (2, 3, 4) use records the model has never seen in training.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/train.csv"
MODEL_PATH = "model.pkl"
UNSEEN_PATH = "data/unseen_data.csv"

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42)),
    ])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing values:\n", df[FEATURES].isna().sum().to_string(), "\n")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)} | Test (unseen) size: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Complete pipeline saved to {MODEL_PATH}")

    # Save unseen records (with true labels, for comparison) for Tasks 2-4
    unseen = X_test.copy()
    unseen[TARGET] = y_test
    unseen.to_csv(UNSEEN_PATH, index=True, index_label="RecordId")
    print(f"Unseen records saved to {UNSEEN_PATH}")


if __name__ == "__main__":
    main()
