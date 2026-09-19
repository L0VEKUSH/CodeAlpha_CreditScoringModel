
# Credit Scoring Intelligence

### CodeAlpha Machine Learning Internship — Task 1

An end-to-end machine learning project that classifies historical credit-risk records using financial and demographic attributes.

The project includes data preprocessing, exploratory data analysis, model training, evaluation, and an interactive Streamlit dashboard.

> Educational demonstration only. This project is not intended for real lending decisions.

---

## Project Overview

Credit risk classification is a machine learning task that uses historical applicant information to estimate whether a record belongs to a good-risk or bad-risk category.

This project uses the German Credit dataset from the UCI Machine Learning Repository.

Three classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest was selected based on five-fold cross-validation ROC-AUC.

---

## Dataset

Source: UCI Machine Learning Repository

Dataset: Statlog (German Credit Data)

Dataset URL:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Dataset characteristics:

- 1,000 records
- 20 input features
- Binary classification target
- 700 good-risk records
- 300 bad-risk records

Target encoding:

| Label | Meaning |
|---|---|
| 0 | Good Credit Risk |
| 1 | Bad Credit Risk |

The original dataset labels were converted from 1/2 to 0/1.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Plotly
- Joblib

---

## Machine Learning Pipeline

1. Dataset loading
2. Exploratory data analysis
3. Target encoding
4. Stratified train-test split
5. Missing-value imputation
6. Categorical feature encoding
7. Numerical feature scaling
8. Model training
9. Five-fold cross-validation
10. Model selection
11. Held-out test evaluation
12. Model serialization
13. Interactive prediction dashboard

Preprocessing is included in the trained scikit-learn Pipeline to prevent test-data leakage.

---

## Model Performance

The dataset was divided into 800 training records and 200 testing records.

### Cross-Validation Results

| Model | CV F1 | CV ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.5696 | 0.7687 |
| Decision Tree | 0.5412 | 0.6842 |
| Random Forest | 0.5832 | 0.7955 |

Random Forest was selected using cross-validation ROC-AUC.

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 73.50% |
| Precision | 54.67% |
| Recall | 68.33% |
| F1-score | 0.6074 |
| ROC-AUC | 0.7996 |

Precision, Recall, and F1-score are reported for the bad-risk class.

The final evaluation was performed on the held-out test set.

These results are specific to this dataset and evaluation procedure.

---

## Dashboard Features

### Credit Risk Prediction

- Interactive applicant input form
- Credit risk classification
- Predicted probability visualization
- Applicant information summary
- Downloadable JSON prediction report

### Model Performance

- Accuracy, Precision, Recall, and F1-score
- ROC-AUC
- Cross-validation model comparison
- Confusion matrix
- ROC curve

### Dataset Insights

- Dataset overview
- Credit risk distribution
- Numerical feature statistics
- Interactive feature exploration

### About Project

- Project objective
- Technology stack
- Machine learning methodology
- Dataset information
- Model limitations

---

## Installation

Clone the repository:

```bash
git clone https://github.com/L0VEKUSH/CodeAlpha_CreditScoringModel.git

cd CodeAlpha_CreditScoringModel
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Model Training

```bash
python src/train.py
```

This command performs model comparison, selects the model using cross-validation, evaluates it on the held-out test set, and saves the trained pipeline.

---

## Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Open:

http://localhost:8501

---

## Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── data/
│   ├── german.data
│   └── credit_data.csv
│
├── notebooks/
│   └── credit_scoring_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── credit_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── app.py
├── requirements.txt
├── requirements-lock.txt
├── README.md
└── .gitignore
```

---

## Limitations

This model is trained on a small historical dataset.

It has not been validated for modern lending populations, probability calibration, fairness, regulatory compliance, or real-world deployment.

The model uses demographic and financial attributes that may raise fairness and legal concerns in real credit decision-making.

Its predictions must not be used to approve, reject, or price actual credit applications.

---

## Internship

**Organization:** CodeAlpha

**Domain:** Machine Learning

**Task:** Credit Scoring Model

**Developer:** Lovekush Kumar

**GitHub:** https://github.com/L0VEKUSH

---

## License and Dataset Attribution

The dataset originates from the UCI Machine Learning Repository.

Review the dataset's licensing and attribution requirements before redistribution.

This project is intended for educational and portfolio demonstration purposes.