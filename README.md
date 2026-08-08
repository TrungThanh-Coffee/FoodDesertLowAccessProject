# Food Desert / Low Food Access Machine Learning Project

## Overview

This project applies and compares three machine learning paradigms to a real-world food-access problem using the **USDA Food Environment Atlas 2020**.

The goal is to identify U.S. counties with a high risk of low food access based on local food environment, food assistance, and socioeconomic indicators.

The project compares:

- **Supervised Learning:** Random Forest
- **Unsupervised Learning:** KMeans
- **Semi-Supervised Learning:** LabelPropagation

It also generates model evaluation metrics, charts, trained model files, county-level risk rankings, and suggested intervention strategies.

---

## Problem Statement

Limited access to healthy and affordable food can affect community health and quality of life.

This project asks:

> Can food-environment and community characteristics be used to identify U.S. counties with a high level of low food access?

The machine learning system is designed as a **screening and decision-support tool**. It does not replace detailed GIS analysis, local surveys, or public-policy evaluation.

---

## Dataset

The project uses selected county-level variables from the **USDA Food Environment Atlas 2020**.

Main data groups include:

- Food store availability
- Grocery and supercenter availability
- Convenience stores
- SNAP and WIC retailers
- Fast-food and full-service restaurants
- Food assistance indicators
- Population characteristics
- Metro / non-metro status
- Persistent poverty indicator

### Main Dataset File

```text
data/food_environment_atlas_2020_selected.csv
```

### Data Dictionary

```text
data/variable_dictionary.csv
```

---

## Target Variable

The original Atlas contains:

```text
PCT_LACCESS_POP10
```

This represents the percentage of the county population with low access to a food store.

For classification, this project creates:

```text
LOW_ACCESS_RISK
```

using:

```python
LOW_ACCESS_RISK = 1 if PCT_LACCESS_POP10 >= 33
LOW_ACCESS_RISK = 0 otherwise
```

Meaning:

- `0` = Normal / Lower Low-Food-Access Risk
- `1` = High Low-Food-Access Risk

> **Important:** The 33% threshold is a project-defined screening threshold for this machine learning assignment. It is not an official USDA definition of a food desert.

---

## Selected Features

### Food Stores

- `GROCPTH11` - Grocery stores per 1,000 population
- `SUPERCPTH11` - Supercenters per 1,000 population
- `CONVSPTH11` - Convenience stores per 1,000 population
- `SPECSPTH11` - Specialized food stores per 1,000 population
- `SNAPSPTH12` - SNAP-authorized stores per 1,000 population
- `WICSPTH11` - WIC-authorized stores per 1,000 population

### Restaurants

- `FFRPTH11` - Fast-food restaurants per 1,000 population
- `FSRPTH11` - Full-service restaurants per 1,000 population

### Food Assistance

- `PC_SNAPBEN12` - SNAP benefits per capita
- `PCT_FREE_LUNCH10` - Percentage eligible for free school lunch

### Community Characteristics

- `PCT_65OLDER10` - Population aged 65 and older
- `PCT_18YOUNGER10` - Population under age 18
- `METRO13` - Metro / non-metro indicator
- `PERPOV10` - Persistent poverty indicator

---

## Data Leakage Prevention

The following variables are excluded from model input:

```text
PCT_LACCESS_POP10
PCT_LACCESS_LOWI10
PCT_LACCESS_HHNV10
```

`PCT_LACCESS_POP10` directly creates the target label.

The other low-access variables are strongly related to the same target concept and could artificially inflate model performance.

---

## Data Preprocessing

The project performs the following preprocessing steps:

1. Load county-level Food Environment Atlas data
2. Preserve five-digit county FIPS codes
3. Replace USDA sentinel values:
   - `-9999` -> missing value
   - `-8888` -> missing value
4. Create the binary target variable
5. Split data into training and testing sets
6. Use stratified splitting to preserve class distribution
7. Fill missing numerical values using the median
8. Fill missing categorical values using the most frequent value
9. Scale numerical features using `StandardScaler`
10. Encode categorical features using `OneHotEncoder`

Train/test split:

```text
80% Training
20% Testing
```

Random seed:

```text
42
```

---

# Machine Learning Algorithms

## 1. Random Forest - Supervised Learning

Random Forest uses fully labeled training data to classify counties as high or lower low-food-access risk.

Main configuration:

```python
RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

Because the dataset is imbalanced, the model uses:

```python
class_weight="balanced"
```

The project also searches for a classification probability threshold using a validation subset inside the training data.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

### Saved Model

```text
models/random_forest.joblib
models/supervised_preprocessor.joblib
```

---

## 2. KMeans - Unsupervised Learning

KMeans clusters counties using only input features.

The algorithm does **not** receive the target label during training.

Configuration:

```python
KMeans(
    n_clusters=2,
    n_init=30,
    random_state=42
)
```

### Evaluation Metrics

- Silhouette Score
- Adjusted Rand Index (ARI)

A post-hoc cluster-to-label mapping is also calculated for explanation only.

### Saved Model

```text
models/kmeans.joblib
models/kmeans_preprocessor.joblib
```

---

## 3. LabelPropagation - Semi-Supervised Learning

LabelPropagation simulates a situation where only a small portion of counties have known labels.

The project keeps approximately:

```text
30% labeled training data
70% unlabeled training data
```

Unknown labels are represented by:

```python
-1
```

Configuration:

```python
LabelPropagation(
    kernel="knn",
    n_neighbors=3,
    max_iter=2000
)
```

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

### Saved Model

```text
models/label_propagation.joblib
```

---

# Project Structure

```text
FoodDesertLowAccessProject/
│
├── project.py
├── requirements.txt
├── README.md
├── SOLUTIONS.md
├── .gitignore
│
├── data/
│   ├── food_environment_atlas_2020_selected.csv
│   └── variable_dictionary.csv
│
├── models/
│   ├── random_forest.joblib
│   ├── supervised_preprocessor.joblib
│   ├── kmeans.joblib
│   ├── kmeans_preprocessor.joblib
│   └── label_propagation.joblib
│
├── outputs/
│   │
│   ├── random_forest/
│   │   ├── 03_random_forest_confusion_matrix.png
│   │   ├── 04_feature_importance.png
│   │   ├── 07_top_county_risk_scores.png
│   │   ├── feature_importance.csv
│   │   ├── random_forest_classification_report.txt
│   │   ├── random_forest_test_predictions.csv
│   │   ├── random_forest_threshold_search.csv
│   │   └── county_risk_priority_and_solutions.csv
│   │
│   ├── kmeans/
│   │   ├── 05_kmeans_pca.png
│   │   ├── kmeans_cluster_vs_label.csv
│   │   ├── kmeans_cluster_mapping.json
│   │   └── clustering_metrics.csv
│   │
│   ├── label_propagation/
│   │   ├── 06_label_propagation_confusion_matrix.png
│   │   └── label_propagation_classification_report.txt
│   │
│   ├── 01_class_distribution.png
│   ├── 02_low_access_distribution.png
│   ├── 08_model_comparison.png
│   ├── classification_metrics.csv
│   ├── data_summary.json
│   ├── missing_values.csv
│   └── run_summary.json
│
└── src/
    └── food_desert_project.py
```

---

# Installation

## Requirements

Recommended:

```text
Python 3.11 or Python 3.12
```

Required Python packages:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Interactive Mode

Run:

```bash
python project.py
```

The program displays:

```text
[1] Random Forest       - Supervised Learning
[2] KMeans              - Unsupervised Learning
[3] LabelPropagation    - Semi-Supervised Learning
[4] Run ALL algorithms
[0] Exit
```

After an algorithm finishes, press `ENTER` to return to the menu.

The application only stops when:

```text
0 - Exit
```

is selected.

---

## Run Random Forest Only

```bash
python project.py --model rf
```

---

## Run KMeans Only

```bash
python project.py --model kmeans
```

---

## Run LabelPropagation Only

```bash
python project.py --model lp
```

---

## Run All Algorithms

```bash
python project.py --model all
```

Running all algorithms also generates model-comparison outputs.

---

# Running in PyCharm

1. Open the entire project folder in PyCharm
2. Configure a Python interpreter
3. Create or select a virtual environment
4. Open the PyCharm Terminal
5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Open:

```text
project.py
```

7. Right-click and select:

```text
Run 'project'
```

---

# Example Terminal Output

```text
==============================================================================
 FOOD DESERT / LOW FOOD ACCESS - MACHINE LEARNING PROJECT
 USDA Food Environment Atlas 2020 | County-level analysis
==============================================================================

>>> SELECT ALGORITHM
------------------------------------------------------------------------------

  [1] Random Forest       - Supervised Learning
  [2] KMeans              - Unsupervised Learning
  [3] LabelPropagation    - Semi-Supervised Learning
  [4] Run ALL algorithms
  [0] Exit
```

Example Random Forest result:

```text
>>> SUPERVISED LEARNING - RANDOM FOREST
------------------------------------------------------------------------------

  Metric                  Score      Percent
  ---------------- ------------ ------------
  Accuracy               0.8251       82.51%
  Precision              0.5588       55.88%
  Recall                 0.3220       32.20%
  F1-score               0.4086       40.86%

  Decision threshold     : 0.525
```

> Exact results may vary slightly depending on Python and scikit-learn versions.

---

# Output Visualizations

The project automatically generates visualizations including:

### Dataset

- Class distribution
- Low food-access percentage distribution

### Random Forest

- Confusion matrix
- Feature importance
- Top county risk scores

### KMeans

- PCA visualization of county clusters

### LabelPropagation

- Confusion matrix

### Comparison

- Random Forest vs LabelPropagation metrics

The figures include short titles, axis labels, legends, and annotations for presentation use.

---

# Model Interpretation

## Random Forest

Best suited when sufficient labeled historical data is available.

It can be used to:

- classify county risk
- rank counties by risk probability
- identify important predictive variables

---

## KMeans

Best suited for exploratory analysis when labels are unavailable.

It can help answer:

> Do counties naturally form groups with similar food-environment characteristics?

A low ARI does not mean KMeans is incorrectly implemented. It means the naturally discovered clusters do not closely match the predefined risk classes.

---

## LabelPropagation

Best suited when labeled data is limited but unlabeled observations are available.

It attempts to transfer label information from labeled counties to similar unlabeled counties.

---

# Practical Application

The Random Forest output includes:

```text
outputs/random_forest/county_risk_priority_and_solutions.csv
```

This file contains:

- County
- State
- FIPS
- Observed low-access percentage
- Project risk label
- Predicted risk probability
- Priority level
- Suggested interventions

---

# Suggested Interventions

## Low Grocery / Supercenter Availability

Possible actions:

- Grocery-store incentives
- Community grocery partnerships
- Farmers' markets
- Mobile markets
- Healthy-food financing

---

## Low SNAP / WIC Retailer Availability

Possible actions:

- Recruit more SNAP-authorized retailers
- Recruit more WIC-authorized retailers
- Support local stores through authorization procedures

---

## High Convenience Store Density

Possible action:

### Healthy Corner Store Program

Possible improvements:

- Fresh fruit
- Vegetables
- Refrigerated products
- Healthy staple foods
- Better supplier partnerships

---

## Rural Counties

Possible actions:

- Mobile grocery routes
- Food delivery programs
- Community pickup hubs
- Transportation partnerships

---

# Recommended Decision Workflow

```text
Food Environment Atlas
        ↓
Machine Learning Screening
        ↓
County Risk Ranking
        ↓
Local GIS / Community Validation
        ↓
Identify Main Access Barrier
        ↓
Select Intervention
        ↓
Pilot Program
        ↓
Measure Results
```

---

# Limitations

This project has several important limitations.

### 1. Project-defined target threshold

The `33%` threshold is created for classification and is not an official USDA food-desert definition.

### 2. County-level aggregation

Food-access problems can vary significantly inside the same county.

Census-tract or neighborhood-level data would provide more precise intervention planning.

### 3. Class imbalance

High-risk counties represent a smaller portion of the dataset.

Therefore, accuracy alone can be misleading.

Precision, recall, and F1-score should also be considered.

### 4. Predictive association is not causation

Random Forest feature importance indicates predictive usefulness.

It does not prove that a feature causes low food access.

### 5. Additional real-world data would improve the system

Future versions could include:

- Travel distance to stores
- Travel time
- Public transportation
- Vehicle availability
- Food prices
- Store opening hours
- Store inventory
- Population density
- Community survey data

---

# Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Threshold optimization focused on recall
- Cross-validation
- SMOTE for supervised experiments
- Additional supervised classifiers
- Improved semi-supervised graph connectivity
- Census-tract-level analysis
- GIS visualization
- Interactive dashboard
- Model explainability using SHAP
- Updated Food Environment Atlas data

---

# Evaluation Focus

For an imbalanced screening problem, the project does not rely only on accuracy.

For supervised and semi-supervised models:

```text
Accuracy
Precision
Recall
F1-score
```

For KMeans:

```text
Silhouette Score
Adjusted Rand Index
```

In practical screening, recall is especially important because a false negative represents a truly high-risk county that the system fails to identify.

---

# Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- joblib
- PyCharm
- Git / GitHub

---

# Authors

Machine Learning Project  
Food Desert / Low Food Access Analysis

---

# Disclaimer

This project is created for educational and machine-learning experimentation purposes.

The generated risk scores and intervention recommendations should not be used as direct public-policy decisions without additional local data, GIS validation, economic analysis, and community consultation.
