import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .config import (
    ARTIFACTS,
    CLUSTER_FEATURES,
    DATA_PATH,
    RANDOM_STATE,
    SCORED_PATH,
)
from .features import engineer_features, load_raw
from .modeling import (
    action_text,
    compute_gpi,
    choose_k,
    evaluate_k,
    label_clusters,
    recommend,
    robustness_report,
    run_dbscan,
    run_hierarchical,
    run_kmeans,
    run_pca,
)
from .preprocess import build_matrix


def lever_importance(num_df, labels):
    labels = pd.Series(np.asarray(labels), index=num_df.index)
    grand = num_df.mean()
    total = ((num_df - grand) ** 2).sum()
    between = sum(
        len(g) * (g.mean() - grand) ** 2 for _, g in num_df.groupby(labels)
    )
    eta2 = (between / total).sort_values(ascending=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        num_df, labels, test_size=0.25, stratify=labels, random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return eta2, float(rf.score(X_te, y_te))


def run(data_path=DATA_PATH, out_path=SCORED_PATH, artifacts=ARTIFACTS):
    artifacts.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data_path)
    feat = engineer_features(raw)

    X, num_df, cat_cols = build_matrix(feat)
    pca, Z, loadings, evr = run_pca(X)

    k_scores = evaluate_k(Z)
    k = choose_k(k_scores)

    km = run_kmeans(Z, k)
    hc = run_hierarchical(Z, k)
    db, eps = run_dbscan(Z)
    robust = robustness_report(km, hc, db, Z, eps)

    mapping, centroids = label_clusters(num_df, km)

    scored = compute_gpi(feat)
    scored["ClusterID"] = km
    scored["ClusterLabel"] = scored["ClusterID"].map(mapping)
    scored["HierarchicalCluster"] = hc
    scored["DBSCANCluster"] = db
    scored["PC1"] = Z[:, 0]
    scored["PC2"] = Z[:, 1]
    scored["PC3"] = Z[:, 2] if Z.shape[1] > 2 else 0.0
    scored = recommend(scored)
    scored["ActionPlan"] = scored.apply(action_text, axis=1)
    scored["GPIRank"] = scored["GPI"].rank(ascending=False, method="first").astype(int)
    scored["GPITier"] = pd.qcut(
        scored["GPI"], 3, labels=["Low", "Medium", "High"]
    ).astype(str)

    imp, rf_acc = lever_importance(num_df, km)

    summary = (
        scored.groupby("ClusterLabel")
        .agg(
            Restaurants=("RestaurantID", "count"),
            GPI=("GPI", "mean"),
            GrowthFactor=("GrowthFactor", "mean"),
            MonthlyOrders=("MonthlyOrders", "mean"),
            AOV=("AOV", "mean"),
            NetMargin=("NetMargin", "mean"),
            CostBurden=("CostBurden", "mean"),
            AggregatorDependence=("AggregatorDependence", "mean"),
            InStoreShare=("InStoreShareCalc", "mean"),
            SelfDeliveryShare=("SelfDeliveryShareCalc", "mean"),
            DeliveryRadiusKM=("DeliveryRadiusKM", "mean"),
            TotalNetProfit=("TotalNetProfit", "mean"),
        )
        .sort_values("GPI", ascending=False)
    )
    rec_mix = pd.crosstab(scored["ClusterLabel"], scored["Recommendation"])

    scored.to_csv(out_path, index=False)
    summary.to_csv(artifacts / "cluster_summary.csv")
    rec_mix.to_csv(artifacts / "cluster_recommendation_mix.csv")
    loadings.to_csv(artifacts / "pca_loadings.csv")
    k_scores.to_csv(artifacts / "k_selection.csv", index=False)
    imp.to_csv(artifacts / "lever_importance.csv", header=["EtaSquared"])
    (centroids.rename(index=mapping)).to_csv(artifacts / "cluster_centroids_zscore.csv")

    metrics = {
        "n_restaurants": int(len(scored)),
        "pca_components": int(Z.shape[1]),
        "pca_variance_explained": [float(v) for v in evr[:6]],
        "pca_cumulative_used": float(evr[: Z.shape[1]].sum()),
        "chosen_k": int(k),
        "cluster_to_label": {str(a): b for a, b in mapping.items()},
        "rf_cluster_accuracy": rf_acc,
        "robustness": robust,
        "recommendation_counts": scored["Recommendation"].value_counts().to_dict(),
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return scored, summary, metrics, loadings, k_scores, imp


if __name__ == "__main__":
    scored, summary, metrics, *_ = run()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(summary.round(3))
    print(json.dumps(metrics, indent=2))
