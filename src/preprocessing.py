
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_data.csv"



def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Complete Step 2 first."
        )

    df = pd.read_csv(DATA_PATH)

    # Check whether the required target column exists.
    if "credit_risk" not in df.columns:
        raise ValueError(
            "Missing target column: credit_risk"
        )

    # Check target values before conversion.
    print("\nOriginal credit risk distribution:")
    print(df["credit_risk"].value_counts(dropna=False))

    if df["credit_risk"].isna().any():
        raise ValueError(
            "credit_risk contains missing values."
        )

    # Ensure labels are numeric.
    df["credit_risk"] = pd.to_numeric(
        df["credit_risk"],
        errors="raise"
    )

    unique_labels = set(df["credit_risk"].unique())

    # German Credit original labels:
    # 1 = Good Risk, 2 = Bad Risk
    if unique_labels == {1, 2}:

        print("\nConverting original labels:")
        print("1 -> 0 (Good Risk)")
        print("2 -> 1 (Bad Risk)")

        df["credit_risk"] = df["credit_risk"].map({
            1: 0,
            2: 1
        })

    # Already converted labels:
    # 0 = Good Risk, 1 = Bad Risk
    elif unique_labels == {0, 1}:

        print("\nTarget labels are already binary.")

    else:

        raise ValueError(
            f"Unexpected credit_risk labels: "
            f"{sorted(unique_labels)}. "
            "Expected either {1, 2} or {0, 1}."
        )

    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"].astype(int)

    print("\nFinal target distribution:")
    print(y.value_counts())

    return X, y


def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


def prepare_data():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test