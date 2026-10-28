import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .config import GPI_WEIGHTS, K_FINAL, K_PREFERRED, K_RANGE, RANDOM_STATE


def run_pca(X, variance=0.90):
    full = PCA(random_state=RANDOM_STATE).fit(X)
    cum = np.cumsum(full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, variance) + 1)
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i + 1}" for i in range(n)],
    )
    return pca, Z, loadings, full.explained_variance_ratio_


def evaluate_k(Z):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Z)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(Z, km.labels_),
                "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def choose_k(scores):
    if K_FINAL is not None:
        return int(K_FINAL)
    pool = scores[scores["k"].isin(K_PREFERRED)]
    return int(pool.sort_values("silhouette", ascending=False).iloc[0]["k"])


def run_kmeans(Z, k):
    return KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(Z).labels_


def run_hierarchical(Z, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z).labels_


def run_dbscan(Z, min_samples=10):
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    dist = np.sort(nn.kneighbors(Z)[0][:, -1])
    eps = float(np.percentile(dist, 90))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(Z).labels_
    return labels, eps


def robustness_report(km_labels, hc_labels, db_labels, Z, eps):
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    return {
        "kmeans_vs_hierarchical_ARI": float(adjusted_rand_score(km_labels, hc_labels)),
        "kmeans_vs_dbscan_ARI": float(adjusted_rand_score(km_labels, db_labels)),
        "dbscan_eps": eps,
        "dbscan_clusters": int(n_clusters),
        "dbscan_noise_share": float((db_labels == -1).mean()),
        "kmeans_silhouette": float(silhouette_score(Z, km_labels)),
        "hierarchical_silhouette": float(silhouette_score(Z, hc_labels)),
    }


def _minmax(s):
    lo, hi = s.min(), s.max()
    if hi - lo == 0:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _pct(s):
    return s.rank(pct=True, method="average")


def compute_gpi(df):
    d = df.copy()

    growth = (
        0.40 * _pct(d["Scale"])
        + 0.35 * _minmax(d["GrowthFactor"])
        + 0.25 * _pct(d["AOV"])
    )
    cost = (
        0.35 * _pct(d["NetMargin"])
        + 0.30 * _pct(1 - d["CostBurden"])
        + 0.20 * _pct(1 - d["CommissionDrag"])
        + 0.15 * _pct(d["RevenueQuality"])
    )
    channel = (
        0.45 * _pct(d["ChannelBalance"])
        + 0.35 * _pct(1 - d["AggregatorDependence"])
        + 0.20 * _pct(d["AggregatorMargin"])
    )
    logistics = (
        0.35 * _pct(d["SelfDeliveryMargin"])
        + 0.30 * _pct(d["ExpansionHeadroom"])
        + 0.20 * _pct(1 - d["DeliveryCostRatio"])
        + 0.15 * _pct(d["DeliveryRadiusKM"])
    )

    d["GrowthSignal"] = growth * 100
    d["CostResilience"] = cost * 100
    d["ChannelBalanceScore"] = channel * 100
    d["LogisticsScalability"] = logistics * 100

    d["GPI"] = sum(w * d[k] for k, w in GPI_WEIGHTS.items())
    d["GPI"] = _minmax(d["GPI"]) * 100
    return d


ARCHETYPES = {
    "High-Growth / High-Risk": {
        "GrowthFactor": 1.0, "Scale": 0.6, "CostBurden": 0.8, "NetMargin": -0.8,
    },
    "Stable Local Performers": {
        "InStoreShareCalc": 1.0, "NetMargin": 0.7, "AggregatorDependence": -0.8,
    },
    "Aggregator-Dependent Low Margin": {
        "AggregatorDependence": 1.0, "CommissionDrag": 0.8, "NetMargin": -1.0,
    },
    "Scalable Self-Delivery Leaders": {
        "SelfDeliveryShareCalc": 1.0, "SelfDeliveryMargin": 0.9, "ExpansionHeadroom": 0.5,
        "DeliveryCostRatio": -0.6,
    },
    "Overextended, Low Return": {
        "DeliveryRadiusKM": 0.7, "CostBurden": 1.0, "NetMargin": -0.8, "GrowthFactor": -0.6,
    },
    "Operationally Strong, Under-Scaled": {
        "NetMargin": 1.0, "RevenueQuality": 0.7, "Scale": -1.0, "CostBurden": -0.5,
    },
}


def label_clusters(num_df, labels):
    centroids = num_df.groupby(labels).mean()
    names = list(ARCHETYPES)
    score = np.zeros((len(centroids), len(names)))
    for j, name in enumerate(names):
        for feat, w in ARCHETYPES[name].items():
            if feat in centroids.columns:
                score[:, j] += w * centroids[feat].to_numpy()
    rows, cols = linear_sum_assignment(-score)
    mapping = {int(centroids.index[r]): names[c] for r, c in zip(rows, cols)}
    return mapping, centroids


def recommend(df):
    d = df.copy()
    hi = d["GPI"].quantile(0.67)
    agg_hi = d["AggregatorDependence"].quantile(0.67)
    comps = ["GrowthSignal", "CostResilience", "ChannelBalanceScore", "LogisticsScalability"]
    weakest = d[comps].idxmin(axis=1)
    d["WeakestDriver"] = weakest
    d["StrongestDriver"] = d[comps].idxmax(axis=1)

    rec = pd.Series("Hold / Stabilize", index=d.index)
    rebalance = (d["AggregatorDependence"] >= agg_hi) | (weakest == "ChannelBalanceScore")
    rec[rebalance & (d["GPI"] < hi)] = "Rebalance channels"
    rec[d["GPI"] >= hi] = "Optimize"
    d["Recommendation"] = rec
    return d


def action_text(row):
    base = {
        "Optimize": "Scale selectively: widen delivery radius and protect margin while demand compounds.",
        "Rebalance channels": "Shift volume from aggregators to in-store and self-delivery; renegotiate commission.",
        "Hold / Stabilize": "Hold expansion; cut cost burden and fix weakest driver before investing.",
    }[row["Recommendation"]]
    focus = {
        "GrowthSignal": "Weakest lever: demand growth and order scale.",
        "CostResilience": "Weakest lever: cost structure and margin.",
        "ChannelBalanceScore": "Weakest lever: channel mix and aggregator exposure.",
        "LogisticsScalability": "Weakest lever: delivery economics and reach.",
    }[row["WeakestDriver"]]
    return f"{base} {focus}"
