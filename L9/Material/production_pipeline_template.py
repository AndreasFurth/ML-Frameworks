"""
The Production Pipeline (Scikit-learn template).

This template combines preprocessing and a model into a single Pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd


def build_pipeline(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=500)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


if __name__ == "__main__":
    # Example usage with a CSV file.
    data = pd.read_csv("data.csv")
    target = "label"

    X = data.drop(columns=[target])
    y = data[target]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))
