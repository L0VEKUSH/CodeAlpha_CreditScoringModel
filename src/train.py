
import sys
import joblib
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate

from preprocessing import (
    prepare_data,
    create_preprocessor
)

from evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_roc_curve,
    save_metrics
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def train_models():

    X_train, X_test, y_train, y_test = prepare_data()

    preprocessor = create_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = {}
    trained_models = {}

    print("\nStarting model comparison...\n")

    for name, estimator in models.items():

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator)
        ])

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "f1": "f1",
                "roc_auc": "roc_auc"
            },
            n_jobs=-1
        )

        results[name] = {
            "CV F1": cv_results["test_f1"].mean(),
            "CV ROC-AUC": cv_results["test_roc_auc"].mean()
        }

        print(
            f"{name}: "
            f"CV F1={results[name]['CV F1']:.4f}, "
            f"CV ROC-AUC={results[name]['CV ROC-AUC']:.4f}"
        )

    comparison = pd.DataFrame(results).T

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv"
    )

    print("\nModel Comparison:")
    print(comparison)

    # Select using training-set cross-validation only.
    # The test set remains untouched during selection.

    best_name = comparison["CV ROC-AUC"].idxmax()

    print(f"\nSelected Model: {best_name}")

    best_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(models[best_name]))
    ])

    best_pipeline.fit(X_train, y_train)

    trained_models[best_name] = best_pipeline

    print("\nFinal held-out test evaluation:")

    final_metrics = evaluate_model(
        best_pipeline,
        X_test,
        y_test,
        best_name
    )

    save_confusion_matrix(
        best_pipeline,
        X_test,
        y_test
    )

    save_roc_curve(
        best_pipeline,
        X_test,
        y_test
    )

    save_metrics({
        "selected_model": best_name,
        "test_metrics": final_metrics
    })

    joblib.dump(
        best_pipeline,
        MODELS_DIR / "credit_model.pkl"
    )

    print("\nModel saved successfully!")
    print("Path: models/credit_model.pkl")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_models()