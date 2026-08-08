from pathlib import Path
import argparse
import os
import sys
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelPropagation

RANDOM_STATE = 42
LOW_ACCESS_THRESHOLD = 33.0
TEST_SIZE = 0.20
LABELED_FRACTION = 0.30

FILE_DIR = Path(__file__).resolve().parent
ROOT = FILE_DIR.parent if FILE_DIR.name == "src" else FILE_DIR
DATA_PATH = ROOT / "data" / "food_environment_atlas_2020_selected.csv"
DICT_PATH = ROOT / "data" / "variable_dictionary.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# Algorithm-specific output folders
RF_OUTPUT_DIR = OUTPUT_DIR / "random_forest"
KMEANS_OUTPUT_DIR = OUTPUT_DIR / "kmeans"
LP_OUTPUT_DIR = OUTPUT_DIR / "label_propagation"

# Create folders automatically if they do not exist.
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RF_OUTPUT_DIR.mkdir(exist_ok=True)
KMEANS_OUTPUT_DIR.mkdir(exist_ok=True)
LP_OUTPUT_DIR.mkdir(exist_ok=True)


# Output structure created by this project:
#
# outputs/
# ├── random_forest/          # Random Forest-only files
# ├── kmeans/                 # KMeans-only files
# ├── label_propagation/      # LabelPropagation-only files
# ├── 01_class_distribution.png
# ├── 02_low_access_distribution.png
# ├── classification_metrics.csv
# ├── 08_model_comparison.png
# ├── data_summary.json
# ├── missing_values.csv
# └── run_summary.json
#
NUMERIC_FEATURES = [
    "GROCPTH11",
    "SUPERCPTH11",
    "CONVSPTH11",
    "SPECSPTH11",
    "SNAPSPTH12",
    "WICSPTH11",
    "FFRPTH11",
    "FSRPTH11",
    "PC_SNAPBEN12",
    "PCT_FREE_LUNCH10",
    "PCT_65OLDER10",
    "PCT_18YOUNGER10",
]

CATEGORICAL_FEATURES = [
    "METRO13",
    "PERPOV10",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_COLUMNS = [
    "PCT_LACCESS_POP10",
    "PCT_LACCESS_LOWI10",
    "PCT_LACCESS_HHNV10",
]


def build_preprocessor():
    """Build a reproducible preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def load_and_prepare_data():
    """Load Atlas data, clean sentinel missing values, and create the ML label."""
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})

    # Preserve leading zeros in the 5-digit county FIPS identifier.
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)

    # USDA Atlas sentinel values -> missing.
    df = df.replace([-9999, -8888], np.nan)

    # Project-defined binary classification label.
    df["LOW_ACCESS_RISK"] = (
        df["PCT_LACCESS_POP10"] >= LOW_ACCESS_THRESHOLD
    ).astype(int)

    return df


def save_data_summary(df):
    """Export concise EDA/data-quality summaries."""
    summary = {
        "rows": int(len(df)),
        "columns_in_source_subset": int(df.shape[1]),
        "low_access_threshold_percent": LOW_ACCESS_THRESHOLD,
        "positive_count": int(df["LOW_ACCESS_RISK"].sum()),
        "negative_count": int((df["LOW_ACCESS_RISK"] == 0).sum()),
        "positive_rate": float(df["LOW_ACCESS_RISK"].mean()),
        "target_mean_pct": float(df["PCT_LACCESS_POP10"].mean()),
        "target_median_pct": float(df["PCT_LACCESS_POP10"].median()),
        "target_75th_percentile_pct": float(df["PCT_LACCESS_POP10"].quantile(0.75)),
    }
    (OUTPUT_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    missing = (
        df[MODEL_FEATURES]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )
    missing["missing_percent"] = missing["missing_count"] / len(df) * 100
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    # Distribution plot
    counts = df["LOW_ACCESS_RISK"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Normal/Lower Risk (0)", "High Low-Access Risk (1)"], counts.values)
    ax.set_title("Class Distribution - Low Food Access Risk")
    ax.set_ylabel("Number of Counties")
    ax.set_xlabel("Risk Class")
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["PCT_LACCESS_POP10"].dropna(), bins=30)
    ax.axvline(
        LOW_ACCESS_THRESHOLD,
        linestyle="--",
        label=f"Project threshold = {LOW_ACCESS_THRESHOLD:.0f}%",
    )
    ax.set_title("Low Food Access Distribution")
    ax.set_xlabel("Population with Low Food Access (%)")
    ax.set_ylabel("Number of Counties")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_low_access_distribution.png", dpi=160)
    plt.close(fig)


def plot_confusion(cm, title, filename, output_dir):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Lower Risk (0)", "High Risk (1)"])
    ax.set_yticks([0, 1], labels=["Lower Risk (0)", "High Risk (1)"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def choose_random_forest_threshold(X_train, y_train):
    """
    Choose a probability threshold on a validation subset inside the training set.

    Why:
      For a public-health / access screening task, the default 0.50 threshold can
      miss many positive counties. We choose the threshold that maximizes F1 on
      validation data only, then evaluate once on the untouched test set.
    """
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    X_subtrain_p = preprocessor.fit_transform(X_subtrain)
    X_val_p = preprocessor.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_subtrain_p, y_subtrain)
    probabilities = model.predict_proba(X_val_p)[:, 1]

    rows = []
    for threshold in np.arange(0.15, 0.601, 0.025):
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_val, pred),
                "precision": precision_score(y_val, pred, zero_division=0),
                "recall": recall_score(y_val, pred, zero_division=0),
                "f1": f1_score(y_val, pred, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(RF_OUTPUT_DIR / "random_forest_threshold_search.csv", index=False)

    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    return float(best_row["threshold"])


def train_supervised(df):
    """Train/evaluate Random Forest on an 80/20 stratified split."""
    X = df[MODEL_FEATURES].copy()
    y = df["LOW_ACCESS_RISK"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    threshold = choose_random_forest_threshold(X_train, y_train)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_p, y_train)

    probabilities = model.predict_proba(X_test_p)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)

    metrics = {
        "Algorithm": "Random Forest",
        "Paradigm": "Supervised",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "DecisionThreshold": threshold,
    }

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(
        cm,
        "Random Forest - Confusion Matrix",
        "03_random_forest_confusion_matrix.png",
        RF_OUTPUT_DIR,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal/Lower Risk", "High Low-Access Risk"],
        zero_division=0,
    )
    (RF_OUTPUT_DIR / "random_forest_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance["feature"] = (
        importance["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    importance.to_csv(RF_OUTPUT_DIR / "feature_importance.csv", index=False)

    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Random Forest - Top Feature Importance")
    ax.set_xlabel("Relative Importance")
    fig.tight_layout()
    fig.savefig(RF_OUTPUT_DIR / "04_feature_importance.png", dpi=160)
    plt.close(fig)

    # Save predictions for the held-out test set.
    test_results = df.loc[X_test.index, ["FIPS", "State", "County", "PCT_LACCESS_POP10"]].copy()
    test_results["actual_label"] = y_test
    test_results["risk_probability"] = probabilities
    test_results["predicted_label"] = y_pred
    test_results.to_csv(RF_OUTPUT_DIR / "random_forest_test_predictions.csv", index=False)

    joblib.dump(preprocessor, MODEL_DIR / "supervised_preprocessor.joblib")
    joblib.dump(model, MODEL_DIR / "random_forest.joblib")

    return metrics, preprocessor, model, threshold, X_train, X_test, y_train, y_test


def train_unsupervised(df):
    """
    KMeans clustering on the same input features.

    KMeans does not see LOW_ACCESS_RISK during training.
    The true labels are used only AFTER clustering for ARI/evaluation.
    """
    X = df[MODEL_FEATURES].copy()
    y = df["LOW_ACCESS_RISK"].to_numpy()

    preprocessor = build_preprocessor()
    X_p = preprocessor.fit_transform(X)

    model = KMeans(
        n_clusters=2,
        n_init=30,
        random_state=RANDOM_STATE,
    )
    clusters = model.fit_predict(X_p)

    silhouette = silhouette_score(X_p, clusters)
    ari = adjusted_rand_score(y, clusters)

    # Binary cluster-to-label mapping for explanatory evaluation only.
    mapped_a = clusters.copy()
    mapped_b = 1 - clusters
    if accuracy_score(y, mapped_b) > accuracy_score(y, mapped_a):
        mapped = mapped_b
        mapping = {"cluster_0": 1, "cluster_1": 0}
    else:
        mapped = mapped_a
        mapping = {"cluster_0": 0, "cluster_1": 1}

    mapping_accuracy = accuracy_score(y, mapped)

    metrics = {
        "Algorithm": "KMeans",
        "Paradigm": "Unsupervised",
        "Silhouette": silhouette,
        "ARI": ari,
        "MappedAccuracy_for_explanation": mapping_accuracy,
    }

    cluster_table = pd.crosstab(
        pd.Series(y, name="ActualLowAccessRisk"),
        pd.Series(clusters, name="Cluster"),
    )
    cluster_table.to_csv(KMEANS_OUTPUT_DIR / "kmeans_cluster_vs_label.csv")

    (KMEANS_OUTPUT_DIR / "kmeans_cluster_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    # PCA is visualization only; KMeans itself used the full feature space.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_p)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, s=15, alpha=0.65)
    ax.set_title("KMeans - County Clusters (PCA Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("KMeans Cluster")
    ax.text(
        0.02, 0.02,
        "Each point = one county\nColors = KMeans clusters",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    fig.tight_layout()
    fig.savefig(KMEANS_OUTPUT_DIR / "05_kmeans_pca.png", dpi=160)
    plt.close(fig)

    joblib.dump(preprocessor, MODEL_DIR / "kmeans_preprocessor.joblib")
    joblib.dump(model, MODEL_DIR / "kmeans.joblib")

    return metrics


def build_stratified_partial_labels(y_train, labeled_fraction):
    """
    Reveal the same fraction of EACH class to avoid accidentally hiding
    nearly all positive labels in this imbalanced problem.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_array = y_train.to_numpy()
    labeled_mask = np.zeros(len(y_array), dtype=bool)

    for class_value in np.unique(y_array):
        indices = np.where(y_array == class_value)[0]
        n_labeled = max(1, int(round(labeled_fraction * len(indices))))
        chosen = rng.choice(indices, size=n_labeled, replace=False)
        labeled_mask[chosen] = True

    y_partial = np.full(len(y_array), -1, dtype=int)
    y_partial[labeled_mask] = y_array[labeled_mask]

    return y_partial, labeled_mask


def train_semisupervised(
    df,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    LabelPropagation:
      - 30% of training labels are visible.
      - 70% are replaced by -1 (unknown).
      - Test labels stay completely hidden during training.
    """
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    y_partial, labeled_mask = build_stratified_partial_labels(
        y_train,
        LABELED_FRACTION,
    )

    model = LabelPropagation(
        kernel="knn",
        n_neighbors=3,
        max_iter=2000,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train_p, y_partial)

    y_pred = model.predict(X_test_p)

    metrics = {
        "Algorithm": "LabelPropagation",
        "Paradigm": "Semi-supervised",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "LabeledFraction": float(labeled_mask.mean()),
    }

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(
        cm,
        "LabelPropagation - Confusion Matrix",
        "06_label_propagation_confusion_matrix.png",
        LP_OUTPUT_DIR,
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal/Lower Risk", "High Low-Access Risk"],
        zero_division=0,
    )
    (LP_OUTPUT_DIR / "label_propagation_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    joblib.dump(model, MODEL_DIR / "label_propagation.joblib")

    return metrics


def solution_recommendation(row, medians, q75):
    """
    Convert descriptive feature patterns into possible intervention ideas.

    IMPORTANT:
      These are decision-support suggestions, NOT causal prescriptions.
      Local GIS, transport, retailer feasibility, cost, and community feedback
      should be checked before intervention.
    """
    recommendations = []

    grocery_gap = (
        pd.notna(row["GROCPTH11"])
        and row["GROCPTH11"] < medians["GROCPTH11"]
    )
    supercenter_gap = (
        pd.notna(row["SUPERCPTH11"])
        and row["SUPERCPTH11"] < medians["SUPERCPTH11"]
    )

    if grocery_gap or supercenter_gap:
        recommendations.append(
            "Expand healthy-food retail capacity: grocery incentives, mobile markets, "
            "farmers' markets, or community grocery partnerships"
        )

    if (
        (pd.notna(row["SNAPSPTH12"]) and row["SNAPSPTH12"] < medians["SNAPSPTH12"])
        or (pd.notna(row["WICSPTH11"]) and row["WICSPTH11"] < medians["WICSPTH11"])
    ):
        recommendations.append(
            "Recruit/support more SNAP- and WIC-authorized food retailers"
        )

    if (
        pd.notna(row["CONVSPTH11"])
        and row["CONVSPTH11"] > medians["CONVSPTH11"]
        and grocery_gap
    ):
        recommendations.append(
            "Healthy-corner-store program: improve fresh produce and staple-food availability "
            "inside existing convenience stores"
        )

    if pd.notna(row["METRO13"]) and int(row["METRO13"]) == 0:
        recommendations.append(
            "Rural-access strategy: mobile grocery routes, delivery/pickup hubs, "
            "and transport partnerships"
        )

    if (
        (pd.notna(row["PC_SNAPBEN12"]) and row["PC_SNAPBEN12"] > q75["PC_SNAPBEN12"])
        or (
            pd.notna(row["PCT_FREE_LUNCH10"])
            and row["PCT_FREE_LUNCH10"] > q75["PCT_FREE_LUNCH10"]
        )
    ):
        recommendations.append(
            "Coordinate retailer-access interventions with SNAP, school-meal, and community food programs"
        )

    if not recommendations:
        recommendations.append(
            "Monitor locally and validate with finer-grained GIS/store-distance data before intervention"
        )

    return " | ".join(recommendations)


def create_priority_and_solution_output(df, preprocessor, model, threshold):
    """
    Rank all counties by Random Forest risk score and attach possible solutions.
    """
    X_all_p = preprocessor.transform(df[MODEL_FEATURES])
    probabilities = model.predict_proba(X_all_p)[:, 1]

    out = df[
        [
            "FIPS",
            "State",
            "County",
            "PCT_LACCESS_POP10",
            "LOW_ACCESS_RISK",
        ]
        + MODEL_FEATURES
    ].copy()

    out["rf_risk_probability"] = probabilities
    out["rf_predicted_high_risk"] = (probabilities >= threshold).astype(int)

    def priority_tier(p):
        if p >= 0.70:
            return "Urgent review"
        if p >= 0.50:
            return "High priority"
        if p >= threshold:
            return "Priority screening"
        return "Lower priority"

    out["priority_tier"] = out["rf_risk_probability"].apply(priority_tier)

    medians = df[MODEL_FEATURES].median(numeric_only=True)
    q75 = df[MODEL_FEATURES].quantile(0.75, numeric_only=True)

    out["recommended_interventions"] = out.apply(
        lambda row: solution_recommendation(row, medians, q75),
        axis=1,
    )

    out = out.sort_values("rf_risk_probability", ascending=False)
    out.to_csv(RF_OUTPUT_DIR / "county_risk_priority_and_solutions.csv", index=False)

    top = out.head(15).copy()
    labels = top["County"] + ", " + top["State"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels[::-1], top["rf_risk_probability"][::-1])
    ax.set_title("Random Forest - Top 15 County Risk Scores")
    ax.set_xlabel("Predicted Probability of High Low-Food-Access Risk")
    ax.text(
        0.99, 0.02,
        "Higher score = higher screening priority",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    fig.tight_layout()
    fig.savefig(RF_OUTPUT_DIR / "07_top_county_risk_scores.png", dpi=160)
    plt.close(fig)


def save_metric_comparison(supervised_metrics, semi_metrics, kmeans_metrics):
    classification_df = pd.DataFrame(
        [
            {
                "Paradigm": supervised_metrics["Paradigm"],
                "Algorithm": supervised_metrics["Algorithm"],
                "Accuracy": supervised_metrics["Accuracy"],
                "Precision": supervised_metrics["Precision"],
                "Recall": supervised_metrics["Recall"],
                "F1": supervised_metrics["F1"],
            },
            {
                "Paradigm": semi_metrics["Paradigm"],
                "Algorithm": semi_metrics["Algorithm"],
                "Accuracy": semi_metrics["Accuracy"],
                "Precision": semi_metrics["Precision"],
                "Recall": semi_metrics["Recall"],
                "F1": semi_metrics["F1"],
            },
        ]
    )
    classification_df.to_csv(
        OUTPUT_DIR / "classification_metrics.csv",
        index=False,
    )

    pd.DataFrame([kmeans_metrics]).to_csv(
        KMEANS_OUTPUT_DIR / "clustering_metrics.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(classification_df))
    width = 0.19
    for index, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(
            x + (index - 1.5) * width,
            classification_df[metric],
            width,
            label=metric,
        )
    ax.set_xticks(x, classification_df["Algorithm"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric Score")
    ax.set_title("Model Comparison - Random Forest vs LabelPropagation")
    ax.legend(title="Evaluation Metric")
    ax.text(
        0.99, 0.02,
        "Higher is better",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_model_comparison.png", dpi=160)
    plt.close(fig)



# ============================================================================
# TERMINAL UI / RUNNER
# ============================================================================

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "dim": "\033[2m",
}


def color(text, name):
    """ANSI color helper for PyCharm/Terminal."""
    return f"{ANSI[name]}{text}{ANSI['reset']}"


def line(char="=", width=78):
    print(color(char * width, "cyan"))


def banner():
    line("=")
    print(color(" FOOD DESERT / LOW FOOD ACCESS - MACHINE LEARNING PROJECT ", "bold"))
    print(color(" USDA Food Environment Atlas 2020 | County-level analysis ", "dim"))
    line("=")


def section(title):
    print()
    print(color(f">>> {title}", "cyan"))
    print(color("-" * 78, "dim"))


def print_dataset_card(df):
    positive = int(df["LOW_ACCESS_RISK"].sum())
    negative = int((df["LOW_ACCESS_RISK"] == 0).sum())

    section("DATASET SUMMARY")
    print(f"  Counties                : {len(df):,}")
    print(f"  Model features          : {len(MODEL_FEATURES)}")
    print(f"  Target                  : LOW_ACCESS_RISK")
    print(f"  High-risk threshold     : >= {LOW_ACCESS_THRESHOLD:.0f}% low-access population")
    print(f"  Lower-risk counties (0) : {negative:,} ({negative / len(df):.2%})")
    print(f"  High-risk counties  (1) : {positive:,} ({positive / len(df):.2%})")
    print(color(
        "  Note: 33% is a project screening threshold, not an official USDA food-desert definition.",
        "yellow",
    ))


def print_classification_metrics(metrics):
    print(f"  {'Metric':<16} {'Score':>12} {'Percent':>12}")
    print(f"  {'-'*16} {'-'*12} {'-'*12}")

    for name in ["Accuracy", "Precision", "Recall", "F1"]:
        label = "F1-score" if name == "F1" else name
        score = metrics[name]
        print(f"  {label:<16} {score:>12.4f} {score:>11.2%}")


def print_rf_result(metrics):
    section("SUPERVISED LEARNING - RANDOM FOREST")
    print(color("  [OK] Training and evaluation completed", "green"))
    print_classification_metrics(metrics)
    print()
    print(f"  Decision threshold      : {metrics['DecisionThreshold']:.3f}")
    print("  Model                   : models/random_forest.joblib")
    print("  Preprocessor            : models/supervised_preprocessor.joblib")
    print()
    print(color(
        "  Interpretation          : Uses labeled data to predict High/Lower Risk counties.",
        "dim",
    ))


def print_kmeans_result(metrics):
    section("UNSUPERVISED LEARNING - KMEANS")
    print(color("  [OK] Clustering and evaluation completed", "green"))
    print(f"  {'Metric':<38} {'Score':>12}")
    print(f"  {'-'*38} {'-'*12}")
    print(f"  {'Silhouette Score':<38} {metrics['Silhouette']:>12.4f}")
    print(f"  {'Adjusted Rand Index (ARI)':<38} {metrics['ARI']:>12.4f}")
    print(
        f"  {'Mapped Accuracy (explanation only)':<38} "
        f"{metrics['MappedAccuracy_for_explanation']:>12.4f}"
    )
    print()
    print("  Model                   : models/kmeans.joblib")
    print("  Preprocessor            : models/kmeans_preprocessor.joblib")
    print()
    print(color(
        "  Interpretation          : Finds natural county groups without using target labels.",
        "dim",
    ))


def print_lp_result(metrics):
    section("SEMI-SUPERVISED LEARNING - LABEL PROPAGATION")
    print(color("  [OK] Training and evaluation completed", "green"))
    print_classification_metrics(metrics)
    print()
    print(f"  Visible training labels : {metrics['LabeledFraction']:.2%}")
    print(f"  Hidden training labels  : {1 - metrics['LabeledFraction']:.2%}")
    print("  Model                   : models/label_propagation.joblib")
    print()
    print(color(
        "  Interpretation          : Learns from a small labeled subset and propagates labels.",
        "dim",
    ))


def print_final_paths():
    section("FINISHED")
    print(color("  [OK] Project run completed successfully", "green"))
    print(f"  Outputs                 : {OUTPUT_DIR}")
    print(f"    Random Forest         : {RF_OUTPUT_DIR}")
    print(f"    KMeans                : {KMEANS_OUTPUT_DIR}")
    print(f"    LabelPropagation      : {LP_OUTPUT_DIR}")
    print(f"  Models                  : {MODEL_DIR}")
    print()



def print_generated_visuals(mode):
    section("GENERATED VISUALS")

    common = [
        ("outputs/01_class_distribution.png", "Class balance"),
        ("outputs/02_low_access_distribution.png", "Low-access distribution"),
    ]

    algorithm_visuals = {
        "rf": [
            ("outputs/random_forest/03_random_forest_confusion_matrix.png",
             "Random Forest confusion matrix"),
            ("outputs/random_forest/04_feature_importance.png",
             "Random Forest feature importance"),
            ("outputs/random_forest/07_top_county_risk_scores.png",
             "Top county risk scores"),
        ],
        "kmeans": [
            ("outputs/kmeans/05_kmeans_pca.png",
             "KMeans clusters in 2D PCA"),
        ],
        "lp": [
            ("outputs/label_propagation/06_label_propagation_confusion_matrix.png",
             "LabelPropagation confusion matrix"),
        ],
    }

    items = list(common)

    if mode == "all":
        items += algorithm_visuals["rf"]
        items += algorithm_visuals["kmeans"]
        items += algorithm_visuals["lp"]
        items += [
            ("outputs/08_model_comparison.png",
             "Random Forest vs LabelPropagation")
        ]
    else:
        items += algorithm_visuals[mode]

    for filename, description in items:
        print(f"  {filename:<72} {description}")


def build_split_for_label_propagation(df):
    """Prepare LabelPropagation so it can be run independently."""
    X = df[MODEL_FEATURES].copy()
    y = df["LOW_ACCESS_RISK"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    return preprocessor, X_train, X_test, y_train, y_test


def choose_from_menu():
    section("SELECT ALGORITHM")
    print("  [1] Random Forest       - Supervised Learning")
    print("  [2] KMeans              - Unsupervised Learning")
    print("  [3] LabelPropagation    - Semi-Supervised Learning")
    print("  [4] Run ALL algorithms")
    print("  [0] Exit")
    print()

    mapping = {
        "1": "rf",
        "2": "kmeans",
        "3": "lp",
        "4": "all",
        "0": "exit",
    }

    while True:
        choice = input(color("  Enter choice [0-4]: ", "bold")).strip()

        if choice in mapping:
            return mapping[choice]

        print(color("  Invalid choice. Enter 0, 1, 2, 3, or 4.", "red"))


def run_selected(mode):
    banner()

    section("LOADING DATA")
    print(f"  Dataset: {DATA_PATH}")

    df = load_and_prepare_data()
    save_data_summary(df)

    print(color("  [OK] Dataset loaded and preprocessing summary generated", "green"))
    print_dataset_card(df)

    run_summary = {
        "project": "Food Desert / Low Food Access Risk",
        "target_source": "PCT_LACCESS_POP10",
        "project_label_definition": (
            f"LOW_ACCESS_RISK = 1 when PCT_LACCESS_POP10 >= "
            f"{LOW_ACCESS_THRESHOLD:.0f}%"
        ),
        "warning": (
            "This is a project-defined county screening label, not an official "
            "USDA food-desert designation."
        ),
        "run_mode": mode,
    }

    if mode == "rf":
        section("RUNNING RANDOM FOREST")
        print("  Training supervised classifier...")

        supervised, preprocessor, rf_model, rf_threshold, X_train, X_test, y_train, y_test = (
            train_supervised(df)
        )

        create_priority_and_solution_output(
            df,
            preprocessor,
            rf_model,
            rf_threshold,
        )

        print_rf_result(supervised)
        run_summary["supervised"] = supervised

    elif mode == "kmeans":
        section("RUNNING KMEANS")
        print("  Clustering counties without using target labels...")

        kmeans = train_unsupervised(df)

        print_kmeans_result(kmeans)
        run_summary["unsupervised"] = kmeans

    elif mode == "lp":
        section("RUNNING LABEL PROPAGATION")
        print(
            f"  Keeping approximately {LABELED_FRACTION:.0%} "
            "of training labels visible..."
        )

        preprocessor, X_train, X_test, y_train, y_test = (
            build_split_for_label_propagation(df)
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )

            semi = train_semisupervised(
                df,
                preprocessor,
                X_train,
                X_test,
                y_train,
                y_test,
            )

        print_lp_result(semi)
        run_summary["semi_supervised"] = semi

    elif mode == "all":
        section("STEP 1/3 - RANDOM FOREST")
        print("  Training supervised classifier...")

        supervised, preprocessor, rf_model, rf_threshold, X_train, X_test, y_train, y_test = (
            train_supervised(df)
        )

        print_rf_result(supervised)

        section("STEP 2/3 - KMEANS")
        print("  Clustering counties without using target labels...")

        kmeans = train_unsupervised(df)

        print_kmeans_result(kmeans)

        section("STEP 3/3 - LABEL PROPAGATION")
        print(
            f"  Keeping approximately {LABELED_FRACTION:.0%} "
            "of training labels visible..."
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )

            semi = train_semisupervised(
                df,
                preprocessor,
                X_train,
                X_test,
                y_train,
                y_test,
            )

        print_lp_result(semi)

        section("MODEL COMPARISON")
        save_metric_comparison(supervised, semi, kmeans)

        print("  Classification metrics  : outputs/classification_metrics.csv")
        print("  KMeans metrics          : outputs/kmeans/clustering_metrics.csv")
        print("  Comparison chart        : outputs/08_model_comparison.png")

        create_priority_and_solution_output(
            df,
            preprocessor,
            rf_model,
            rf_threshold,
        )

        run_summary["supervised"] = supervised
        run_summary["unsupervised"] = kmeans
        run_summary["semi_supervised"] = semi

    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    print_generated_visuals(mode)
    print_final_paths()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Food Desert / Low Food Access Machine Learning Project"
    )

    parser.add_argument(
        "--model",
        choices=["rf", "kmeans", "lp", "all"],
        help=(
            "Run one algorithm directly. "
            "rf=Random Forest, kmeans=KMeans, lp=LabelPropagation, all=all. "
            "If omitted, an interactive menu is shown."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Direct command-line mode:
    # python project.py --model rf
    # python project.py --model kmeans
    # python project.py --model lp
    # python project.py --model all
    if args.model:
        run_selected(args.model)
        return

    # Interactive mode: keep returning to menu until Exit is selected.
    while True:
        banner()
        mode = choose_from_menu()

        if mode == "exit":
            print()
            line("=")
            print(color(" Program closed. Thank you! ", "yellow"))
            line("=")
            break

        run_selected(mode)

        print()
        input(color("Press ENTER to return to the main menu...", "bold"))
        print()
        print()


if __name__ == "__main__":
    main()