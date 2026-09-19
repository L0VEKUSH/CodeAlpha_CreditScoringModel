
import json
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        )
    }

    print(f"\n===== {model_name} =====")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Good Risk", "Bad Risk"],
            zero_division=0
        )
    )

    return metrics


def save_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Good Risk", "Bad Risk"]
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title("Credit Scoring Confusion Matrix")

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)


def save_roc_curve(model, X_test, y_test):

    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    ax.set_title("Credit Scoring ROC Curve")

    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close(fig)


def save_metrics(metrics):

    with open(
        RESULTS_DIR / "model_metrics.json",
        "w"
    ) as file:

        json.dump(metrics, file, indent=4)