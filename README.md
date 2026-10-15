# Cardiovascular Disease Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

An end-to-end Machine Learning project to predict the presence or absence of cardiovascular disease (CVD) based on clinical examination findings, demographic attributes, and lifestyle biomarkers.

---

## 1. Project Overview & Clinical Context
Cardiovascular diseases (CVDs) remain the leading cause of premature mortality worldwide, claiming approximately 17.9 million lives each year (representing ~32% of all global fatalities). Early detection and non-invasive risk stratification allow clinicians to introduce targeted lifestyle and pharmacological interventions before catastrophic cardiac events (e.g., myocardial infarction or stroke) occur.

This project analyzes a cohort of **70,000 patient records** from `cardio_train.csv` to construct a predictive diagnostic pipeline using five machine learning algorithms: **Logistic Regression, Support Vector Machine (LinearSVC with calibrated probabilities), K-Nearest Neighbors, Decision Tree, and Random Forest**.

---

## 2. Dataset Information
- **Filename:** `cardio_train.csv` (semicolon-delimited `;`)
- **Total Samples:** 70,000 patient examination records
- **Target Variable:** `cardio` (Binary: `0` = Healthy / No CVD, `1` = Cardiovascular Disease Present)
- **Class Distribution:** Balanced (~50.2% class 0, ~49.8% class 1)

### Features Description:
| Column | Type | Unit / Encoding | Clinical Description |
| :--- | :---: | :---: | :--- |
| `id` | Identifier | Integer | Unique patient identifier (dropped during preprocessing) |
| `age` | Numerical | Days | Age of patient at examination (converted to years: `age / 365.25`) |
| `gender` | Categorical | `1`: Female, `2`: Male | Patient biological sex |
| `height` | Numerical | Centimeters (cm) | Stature of the patient |
| `weight` | Numerical | Kilograms (kg) | Body mass |
| `ap_hi` | Numerical | mmHg | Systolic blood pressure |
| `ap_lo` | Numerical | mmHg | Diastolic blood pressure |
| `cholesterol` | Categorical | `1`: Normal, `2`: Above Normal, `3`: Well Above Normal | Total serum cholesterol category |
| `gluc` | Categorical | `1`: Normal, `2`: Above Normal, `3`: Well Above Normal | Fasting blood glucose category |
| `smoke` | Binary | `0`: Non-smoker, `1`: Smoker | Tobacco smoking status |
| `alco` | Binary | `0`: Non-drinker, `1`: Drinker | Alcohol intake status |
| `active` | Binary | `0`: Sedentary, `1`: Active | Physical activity status |
| `cardio` | Binary | `0`: Healthy, `1`: Disease | Target: Cardiovascular disease presence |

---

## 3. End-to-End Pipeline & Methodology

```
+-------------------------------------------------------+
| 1. Data Ingestion & Audit                             |
|    - Schema, types, 0 nulls, duplicate identification |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 2. Preprocessing & Feature Engineering                |
|    - Drop 'id', Age (days -> years), Derive BMI       |
|    - Filter physiological outliers (BP, Height, etc.) |
|    - Retained 68,518 valid records (97.88%)           |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 3. Exploratory Data Analysis (EDA)                    |
|    - 10 targeted visualization figures                |
|    - Dedicated clinical markdown insights under each  |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 4. Correlation & Multicollinearity Analysis           |
|    - Pearson heatmap, risk association decomposition  |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 5. Machine Learning Modeling (80/20 Stratified Split) |
|    - LR, LinearSVC (Calibrated), KNN, DT, RF          |
|    - Accuracy, Precision, Recall, F1, Confusion Matrix|
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 6. Comparison & ROC Analysis                          |
|    - Unified benchmarking table, bar chart, ROC-AUC   |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 7. Best Model Tuning & Feature Importances            |
|    - RandomizedSearchCV (3-fold CV)                   |
|    - Feature importances extraction                   |
|    - Exported: 'cardio_best_model.joblib' &           |
|      'best_model.pkl'                                 |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 8. Single-Patient Clinical Inference Engine           |
|    - predict_cardio_risk() interactive diagnostic card|
+-------------------------------------------------------+
```

---

## 4. Benchmark Performance & Results Table

All models were evaluated on the held-out test set ($N = 13,704$ patients) under identical stratified conditions (`random_state=42`, `stratify=y`):

| Model | Test Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **74.01%** | **75.40%** | **71.30%** | **73.29%** | **0.7984** |
| **Decision Tree** | 73.20% | 74.88% | 69.83% | 72.27% | 0.7891 |
| **K-Nearest Neighbors (KNN)** | 72.97% | 73.91% | 70.93% | 72.39% | 0.7780 |
| **Logistic Regression** | 72.92% | 74.82% | 68.91% | 71.74% | 0.7887 |
| **Support Vector Machine (LinearSVC)** | 72.90% | 74.95% | 68.64% | 71.66% | 0.7885 |

### Winning Model Justification:
The **Random Forest Classifier** is the selected best model:
1. **Highest Predictive Accuracy:** Achieves **74.01%** generalization accuracy on unseen test data.
2. **Superior Discrimination (ROC-AUC 0.7984):** Dominates all competitor models across every false-positive rate threshold.
3. **Robust Non-linear Modeling:** Captures clinical threshold interactions (e.g. synergistic impact of hypertension + dyslipidemia + age) without suffering from single-tree overfitting.
4. **Primary Clinical Drivers:** Feature importance analysis confirms that blood pressure (`ap_hi` at ~36.8% and `ap_lo` at ~17.8%), age (`age_years` at ~14.4%), BMI (~8.1%), and cholesterol (~8.0%) constitute the decisive diagnostic factors.

---

## 5. Project Directory & Deliverables

| File | Format | Description |
| :--- | :---: | :--- |
| `cardio_project.ipynb` | Jupyter Notebook | Primary interactive notebook with all 64 cells executed and outputs/plots saved |
| `cardio_project.html` | Web / HTML | Complete standalone HTML export for web viewing |
| `cardio_project.pdf` | Document / PDF | Clean compiled report export for offline review |
| `best_model.pkl` | Pickle / Joblib | Serialized winning Random Forest model ready for production |
| `cardio_best_model.joblib` | Joblib Payload | Full serialized model payload with fitted scaler and metadata |
| `cardio_train.csv` | CSV Dataset | Raw clinical dataset (70,000 records) |
| `README.md` | Markdown | Comprehensive project documentation and benchmark guide |
| `viva_notes.md` | Markdown | In-depth viva voce explanation and top 10 examination Q&As |
| `build_cardio_notebook.py` | Python Script | Automated runner script to build and execute the notebook top-to-bottom |

---

## 6. How to Run the Project

### Prerequisites:
- Python 3.10+ (tested on Python 3.14)
- Packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `nbformat`, `nbclient`, `ipykernel`

### Step 1: Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib nbformat nbclient ipykernel
```

### Step 2: Open and Run the Jupyter Notebook
```bash
jupyter notebook cardio_project.ipynb
```
Select **Kernel -> Restart & Run All** to re-run all 64 cells top to bottom.

### Step 3: Run Single-Patient Inference Demo
Within Python or inside the notebook:
```python
from cardio_project import predict_cardio_risk  # or run Cell 8.1 in the notebook

patient = {
    'age_years': 55.0,
    'gender': 2,        # Male
    'height': 175,
    'weight': 88.0,
    'ap_hi': 145,       # Stage 2 Hypertension
    'ap_lo': 92,
    'cholesterol': 3,   # Well above normal
    'gluc': 1,
    'smoke': 1,
    'alco': 0,
    'active': 0
}

risk = predict_cardio_risk(patient, model_path='cardio_best_model.joblib')
```
