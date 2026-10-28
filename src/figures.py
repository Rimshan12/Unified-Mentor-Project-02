import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ARTIFACTS, FIGURES, SCORED_PATH

PALETTE = ["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#C62828", "#00838F"]
COMPONENTS = ["GrowthSignal", "CostResilience", "ChannelBalanceScore", "LogisticsScalability"]
COMP_LABELS = ["Growth signal", "Cost resilience", "Channel balance", "Logistics scalability"]
CHANNELS = ["InStore", "UberEats", "DoorDash", "SelfDelivery"]


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def make_all():
    sns.set_theme(style="whitegrid", font_scale=0.95)
    d = pd.read_csv(SCORED_PATH)
    order = d.groupby("ClusterLabel")["GPI"].mean().sort_values(ascending=False).index.tolist()
    colors = dict(zip(order, PALETTE))

    margins = [d[f"{c}NetProfit"].sum() / d[f"{c}Revenue"].sum() * 100 for c in CHANNELS]
    neg = [(d[f"{c}NetProfit"] < 0).mean() * 100 for c in CHANNELS]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(CHANNELS, margins, color=PALETTE[:4])
    ax[0].set_title("Net margin by channel (%)")
    ax[1].bar(CHANNELS, neg, color=PALETTE[:4])
    ax[1].set_title("Restaurants losing money in channel (%)")
    _save(fig, "01_channel_economics.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.boxplot(data=d, x="Segment", y="NetMargin", hue="Segment", palette=PALETTE[:4], legend=False, ax=ax[0])
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_title("Net margin by segment")
    sns.boxplot(data=d, x="Segment", y="CostBurden", hue="Segment", palette=PALETTE[:4], legend=False, ax=ax[1])
    ax[1].set_title("COGS + OPEX rate by segment")
    _save(fig, "02_segment_cost_margin.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sub = d.groupby("Subregion")[["GrowthFactor", "GPI"]].mean().sort_values("GrowthFactor")
    ax.bar(sub.index, sub["GrowthFactor"], color=PALETTE[:4])
    ax.set_ylim(0.97, 1.06)
    ax.set_title("Mean growth factor by subregion")
    _save(fig, "03_growth_by_subregion.png")

    load = pd.read_csv(ARTIFACTS / "pca_loadings.csv", index_col=0)
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    evr = metrics["pca_variance_explained"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(range(1, len(evr) + 1), np.array(evr) * 100, color=PALETTE[0])
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, marker="o", color=PALETTE[2])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA variance explained (bars) and cumulative (line)")
    _save(fig, "04_pca_variance.png")

    top = load[["PC1", "PC2", "PC3"]]
    idx = top.abs().max(axis=1).sort_values(ascending=False).head(14).index
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(top.loc[idx], annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar=False)
    ax.set_title("Strongest PCA loadings")
    _save(fig, "05_pca_loadings.png")

    k = pd.read_csv(ARTIFACTS / "k_selection.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(k["k"], k["inertia"], marker="o", color=PALETTE[0])
    ax[0].set_title("Elbow (inertia)")
    ax[0].set_xlabel("k")
    ax[1].plot(k["k"], k["silhouette"], marker="o", color=PALETTE[1], label="Silhouette")
    ax2 = ax[1].twinx()
    ax2.plot(k["k"], k["davies_bouldin"], marker="s", color=PALETTE[4], label="Davies-Bouldin")
    ax[1].set_title("Silhouette and Davies-Bouldin")
    ax[1].set_xlabel("k")
    _save(fig, "06_k_selection.png")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name in order:
        s = d[d["ClusterLabel"] == name]
        ax.scatter(s["PC1"], s["PC2"], s=16, alpha=0.7, color=colors[name], label=name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Restaurant cluster map (PCA projection)")
    ax.legend(fontsize=8, loc="best")
    _save(fig, "07_cluster_map.png")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.boxplot(data=d, y="ClusterLabel", x="GPI", order=order, hue="ClusterLabel", palette=colors, legend=False, ax=ax)
    ax.set_title("Growth Potential Index by cluster")
    ax.set_ylabel("")
    _save(fig, "08_gpi_by_cluster.png")

    comp = d.groupby("ClusterLabel")[COMPONENTS].mean().loc[order]
    ang = np.linspace(0, 2 * np.pi, len(COMPONENTS), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.2, 6.2), subplot_kw={"polar": True})
    for name in order:
        vals = comp.loc[name].tolist() + [comp.loc[name].iloc[0]]
        ax.plot(ang, vals, color=colors[name], label=name, lw=1.8)
        ax.fill(ang, vals, color=colors[name], alpha=0.06)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(COMP_LABELS)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=2)
    ax.set_title("GPI components by cluster")
    _save(fig, "09_cluster_radar.png")

    mix = pd.crosstab(d["ClusterLabel"], d["Recommendation"]).loc[order]
    fig, ax = plt.subplots(figsize=(9, 4))
    mix.plot(kind="barh", stacked=True, ax=ax, color={"Optimize": "#2E7D32", "Rebalance channels": "#EF6C00", "Hold / Stabilize": "#546E7A"})
    ax.set_ylabel("")
    ax.set_title("Strategic recommendation mix by cluster")
    ax.invert_yaxis()
    _save(fig, "10_recommendation_mix.png")

    lev = pd.read_csv(ARTIFACTS / "lever_importance.csv", index_col=0)["EtaSquared"].sort_values().tail(14)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(lev.index, lev.values, color=PALETTE[0])
    ax.set_title("Levers separating the archetypes (eta squared)")
    _save(fig, "11_lever_importance.png")


if __name__ == "__main__":
    make_all()
