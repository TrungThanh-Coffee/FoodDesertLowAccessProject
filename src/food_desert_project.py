"""
Food Desert / Low Food Access Machine Learning Project
======================================================

Problem:
    Identify U.S. counties at HIGH RISK of low food access using the USDA
    Food Environment Atlas 2020 county-level indicators.

Important terminology:
    The Food Environment Atlas contains PCT_LACCESS_POP10, the percentage of
    people in a county with low access to a food store in 2010.
    This project creates an ML classification label:

        LOW_ACCESS_RISK = 1 if PCT_LACCESS_POP10 >= 33%
                          0 otherwise

    The 33% cutoff is a PROJECT-DEFINED screening threshold for classification.
    It is NOT an official USDA county-level "food desert" designation.

Assignment paradigms:
    1. Supervised: Random Forest
    2. Unsupervised: KMeans
    3. Semi-supervised: LabelPropagation

Evaluation:
    Random Forest / LabelPropagation:
        Accuracy, Precision, Recall, F1
    KMeans:
        Silhouette Score, Adjusted Rand Index (ARI)

The project intentionally EXCLUDES:
    - PCT_LACCESS_POP10 from X because it creates the target.
    - PCT_LACCESS_LOWI10 and PCT_LACCESS_HHNV10 because they are derived
      from the same low-access concept and would create strong target leakage.

Run:
    python project.py
"""

from pathlib import Path
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

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "food_environment_atlas_2020_selected.csv"
DICT_PATH = ROOT / "data" / "variable_dictionary.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

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
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Counties")
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
    ax.set_title("Distribution of County Low Food Access Percentage")
    ax.set_xlabel("PCT_LACCESS_POP10 (%)")
    ax.set_ylabel("Number of Counties")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_low_access_distribution.png", dpi=160)
    plt.close(fig)


def plot_confusion(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
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
    threshold_df.to_csv(OUTPUT_DIR / "random_forest_threshold_search.csv", index=False)

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
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal/Lower Risk", "High Low-Access Risk"],
        zero_division=0,
    )
    (OUTPUT_DIR / "random_forest_classification_report.txt").write_text(
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
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_feature_importance.png", dpi=160)
    plt.close(fig)

    # Save predictions for the held-out test set.
    test_results = df.loc[X_test.index, ["FIPS", "State", "County", "PCT_LACCESS_POP10"]].copy()
    test_results["actual_label"] = y_test
    test_results["risk_probability"] = probabilities
    test_results["predicted_label"] = y_pred
    test_results.to_csv(OUTPUT_DIR / "random_forest_test_predictions.csv", index=False)

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
    cluster_table.to_csv(OUTPUT_DIR / "kmeans_cluster_vs_label.csv")

    (OUTPUT_DIR / "kmeans_cluster_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    # PCA is visualization only; KMeans itself used the full feature space.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_p)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, s=15, alpha=0.65)
    ax.set_title("KMeans Clusters Projected to 2D with PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_kmeans_pca.png", dpi=160)
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
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal/Lower Risk", "High Low-Access Risk"],
        zero_division=0,
    )
    (OUTPUT_DIR / "label_propagation_classification_report.txt").write_text(
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
    out.to_csv(OUTPUT_DIR / "county_risk_priority_and_solutions.csv", index=False)

    top = out.head(15).copy()
    labels = top["County"] + ", " + top["State"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels[::-1], top["rf_risk_probability"][::-1])
    ax.set_title("Top 15 Counties by Model Risk Score")
    ax.set_xlabel("Predicted High Low-Access Risk Probability")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_top_county_risk_scores.png", dpi=160)
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
        OUTPUT_DIR / "clustering_metrics.csv",
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
    ax.set_ylabel("Score")
    ax.set_title("Supervised vs Semi-Supervised Classification Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_model_comparison.png", dpi=160)
    plt.close(fig)


def main():
    print("=" * 72)
    print("FOOD DESERT / LOW FOOD ACCESS MACHINE LEARNING PROJECT")
    print("=" * 72)

    df = load_and_prepare_data()
    save_data_summary(df)

    print(f"Counties: {len(df):,}")
    print(
        f"High-risk label rate: {df['LOW_ACCESS_RISK'].mean():.2%} "
        f"(threshold >= {LOW_ACCESS_THRESHOLD:.0f}% low-access population)"
    )
    print()

    supervised, preprocessor, rf_model, rf_threshold, X_train, X_test, y_train, y_test = (
        train_supervised(df)
    )
    print("Random Forest:")
    print(supervised)
    print()

    kmeans = train_unsupervised(df)
    print("KMeans:")
    print(kmeans)
    print()

    semi = train_semisupervised(
        df,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print("LabelPropagation:")
    print(semi)
    print()

    save_metric_comparison(supervised, semi, kmeans)
    create_priority_and_solution_output(
        df,
        preprocessor,
        rf_model,
        rf_threshold,
    )

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
        "supervised": supervised,
        "unsupervised": kmeans,
        "semi_supervised": semi,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Selected Random Forest probability threshold: {rf_threshold:.3f}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Models:  {MODEL_DIR}")


if __name__ == "__main__":
    main()
