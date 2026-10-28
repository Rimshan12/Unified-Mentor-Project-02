import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ARTIFACTS, SCORED_PATH
from src.pipeline import run

st.set_page_config(page_title="SkyCity Growth Intelligence", layout="wide")

COMPONENTS = ["GrowthSignal", "CostResilience", "ChannelBalanceScore", "LogisticsScalability"]
COMPONENT_NAMES = {
    "GrowthSignal": "Growth signal",
    "CostResilience": "Cost resilience",
    "ChannelBalanceScore": "Channel balance",
    "LogisticsScalability": "Logistics scalability",
}
KPI_COLS = {
    "Scale": "Scale",
    "CostDiscipline": "Cost discipline",
    "AggregatorDependence": "Aggregator dependence",
    "ExpansionHeadroom": "Expansion headroom",
    "RevenueQuality": "Revenue quality",
}
REC_COLORS = {
    "Optimize": "#2E7D32",
    "Rebalance channels": "#EF6C00",
    "Hold / Stabilize": "#546E7A",
}


@st.cache_data(show_spinner="Running analysis pipeline...")
def load():
    if not SCORED_PATH.exists():
        run()
    df = pd.read_csv(SCORED_PATH)
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    summary = pd.read_csv(ARTIFACTS / "cluster_summary.csv", index_col=0)
    levers = pd.read_csv(ARTIFACTS / "lever_importance.csv", index_col=0)
    return df, metrics, summary, levers


def kpi_percentiles(frame, full):
    out = pd.DataFrame(index=frame.index)
    for col in KPI_COLS:
        ranks = full[col].rank(pct=True)
        vals = ranks.loc[frame.index]
        out[col] = (1 - vals) * 100 if col == "AggregatorDependence" else vals * 100
    return out


def radar(series_map, categories, title):
    fig = go.Figure()
    for name, vals in series_map.items():
        v = list(vals) + [vals[0]]
        fig.add_trace(go.Scatterpolar(r=v, theta=categories + [categories[0]], fill="toself", name=name))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], visible=True)),
        title=title,
        height=460,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


df, metrics, summary, levers = load()

st.title("SkyCity Auckland Restaurants & Bars")
st.caption("Restaurant Growth Potential Modeling & Strategic Classification System")

with st.sidebar:
    st.header("Filters")
    sub = st.multiselect("Subregion", sorted(df["Subregion"].unique()), default=sorted(df["Subregion"].unique()))
    cui = st.multiselect("Cuisine", sorted(df["CuisineType"].unique()), default=sorted(df["CuisineType"].unique()))
    seg = st.multiselect("Segment", sorted(df["Segment"].unique()), default=sorted(df["Segment"].unique()))
    clu = st.multiselect("Cluster", sorted(df["ClusterLabel"].unique()), default=sorted(df["ClusterLabel"].unique()))
    gpi_min, gpi_max = st.slider("GPI range", 0.0, 100.0, (0.0, 100.0))

mask = (
    df["Subregion"].isin(sub)
    & df["CuisineType"].isin(cui)
    & df["Segment"].isin(seg)
    & df["ClusterLabel"].isin(clu)
    & df["GPI"].between(gpi_min, gpi_max)
)
view = df[mask].copy()

if view.empty:
    st.warning("No restaurants match the current filters.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Restaurants", f"{len(view):,}")
c2.metric("Avg GPI", f"{view['GPI'].mean():.1f}")
c3.metric("Avg net margin", f"{view['NetMargin'].mean() * 100:.1f}%")
c4.metric("Aggregator dependence", f"{view['AggregatorDependence'].mean() * 100:.1f}%")
c5.metric("Ready to optimize", f"{(view['Recommendation'] == 'Optimize').mean() * 100:.0f}%")

tab_map, tab_score, tab_radar, tab_strategy, tab_compare, tab_drivers = st.tabs(
    ["Cluster map", "Growth scorecards", "Radar charts", "Strategy panel", "Compare clusters", "Growth drivers"]
)

with tab_map:
    axes = st.radio("Projection", ["PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3"], horizontal=True)
    x, y = {"PC1 vs PC2": ("PC1", "PC2"), "PC1 vs PC3": ("PC1", "PC3"), "PC2 vs PC3": ("PC2", "PC3")}[axes]
    fig = px.scatter(
        view,
        x=x,
        y=y,
        color="ClusterLabel",
        size="GPI",
        size_max=14,
        hover_name="RestaurantName",
        hover_data={"GPI": ":.1f", "Segment": True, "CuisineType": True, "Subregion": True, x: False, y: False},
        height=620,
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    evr = metrics["pca_variance_explained"]
    st.caption(
        f"PCA retained {metrics['pca_components']} components "
        f"({metrics['pca_cumulative_used'] * 100:.1f}% variance). PC1 explains {evr[0] * 100:.1f}%, "
        f"PC2 {evr[1] * 100:.1f}%, PC3 {evr[2] * 100:.1f}%."
    )

with tab_score:
    left, right = st.columns([2, 1])
    with left:
        top_n = st.slider("Show top N restaurants by GPI", 5, 50, 15)
        cols = ["GPIRank", "RestaurantName", "CuisineType", "Segment", "Subregion", "ClusterLabel", "GPI"] + COMPONENTS + ["Recommendation"]
        table = view.sort_values("GPI", ascending=False).head(top_n)[cols].rename(columns=COMPONENT_NAMES)
        st.dataframe(
            table.style.format({"GPI": "{:.1f}", **{v: "{:.0f}" for v in COMPONENT_NAMES.values()}}),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        hist = px.histogram(view, x="GPI", color="Recommendation", nbins=30, color_discrete_map=REC_COLORS, height=380)
        hist.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(hist, use_container_width=True)
    avg = view.groupby("ClusterLabel")[COMPONENTS].mean().rename(columns=COMPONENT_NAMES)
    heat = px.imshow(avg.round(1), text_auto=True, aspect="auto", color_continuous_scale="RdYlGn", zmin=0, zmax=100, height=380)
    st.plotly_chart(heat, use_container_width=True)

with tab_radar:
    mode = st.radio("Radar view", ["Restaurants", "Clusters"], horizontal=True)
    cats_gpi = [COMPONENT_NAMES[c] for c in COMPONENTS]
    cats_kpi = list(KPI_COLS.values())
    if mode == "Restaurants":
        options = view.sort_values("GPI", ascending=False)
        labels = options.apply(lambda r: f"{r['RestaurantName']} (ID {r['RestaurantID']})", axis=1)
        lookup = dict(zip(labels, options.index))
        picks = st.multiselect("Pick up to 4 restaurants", list(lookup), default=list(lookup)[:2], max_selections=4)
        if picks:
            idx = [lookup[p] for p in picks]
            sub_df = df.loc[idx]
            pct = kpi_percentiles(sub_df, df)
            a, b = st.columns(2)
            a.plotly_chart(radar({p: sub_df.loc[lookup[p], COMPONENTS].tolist() for p in picks}, cats_gpi, "GPI components"), use_container_width=True)
            b.plotly_chart(radar({p: pct.loc[lookup[p], list(KPI_COLS)].tolist() for p in picks}, cats_kpi, "KPI percentile (higher is better)"), use_container_width=True)
    else:
        grp = view.groupby("ClusterLabel")
        comp = grp[COMPONENTS].mean()
        kp = kpi_percentiles(view, df).groupby(view["ClusterLabel"]).mean()
        a, b = st.columns(2)
        a.plotly_chart(radar({n: comp.loc[n].tolist() for n in comp.index}, cats_gpi, "GPI components by cluster"), use_container_width=True)
        b.plotly_chart(radar({n: kp.loc[n, list(KPI_COLS)].tolist() for n in kp.index}, cats_kpi, "KPI percentile by cluster"), use_container_width=True)

with tab_strategy:
    a, b = st.columns([1, 2])
    counts = view["Recommendation"].value_counts().reset_index()
    counts.columns = ["Recommendation", "Restaurants"]
    a.plotly_chart(
        px.pie(counts, names="Recommendation", values="Restaurants", color="Recommendation", color_discrete_map=REC_COLORS, hole=0.45, height=360),
        use_container_width=True,
    )
    mix = pd.crosstab(view["ClusterLabel"], view["Recommendation"]).reset_index().melt(id_vars="ClusterLabel", var_name="Recommendation", value_name="Restaurants")
    b.plotly_chart(
        px.bar(mix, x="Restaurants", y="ClusterLabel", color="Recommendation", orientation="h", color_discrete_map=REC_COLORS, height=360),
        use_container_width=True,
    )
    st.subheader("Restaurant drill-down")
    names = view.sort_values("GPI", ascending=False)
    labels = names.apply(lambda r: f"{r['RestaurantName']} (ID {r['RestaurantID']})", axis=1).tolist()
    choice = st.selectbox("Select a restaurant", labels)
    row = names.iloc[labels.index(choice)]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GPI", f"{row['GPI']:.1f}", f"rank {int(row['GPIRank'])} of {len(df)}")
    m2.metric("Cluster", row["ClusterLabel"])
    m3.metric("Recommendation", row["Recommendation"])
    m4.metric("Net margin", f"{row['NetMargin'] * 100:.1f}%")
    st.info(row["ActionPlan"])
    contrib = pd.DataFrame(
        {
            "Driver": [COMPONENT_NAMES[c] for c in COMPONENTS],
            "This restaurant": [row[c] for c in COMPONENTS],
            "Cluster average": [df.loc[df["ClusterLabel"] == row["ClusterLabel"], c].mean() for c in COMPONENTS],
        }
    ).melt(id_vars="Driver", var_name="Series", value_name="Value")
    st.plotly_chart(px.bar(contrib, x="Driver", y="Value", color="Series", barmode="group", range_y=[0, 100], height=380), use_container_width=True)
    st.download_button(
        "Download filtered results (CSV)",
        view[["RestaurantID", "RestaurantName", "CuisineType", "Segment", "Subregion", "ClusterLabel", "GPI", "Recommendation", "ActionPlan"]].to_csv(index=False),
        file_name="skycity_strategy_output.csv",
        mime="text/csv",
    )

with tab_compare:
    prof = view.groupby("ClusterLabel").agg(
        Restaurants=("RestaurantID", "count"),
        GPI=("GPI", "mean"),
        NetMargin=("NetMargin", "mean"),
        CostBurden=("CostBurden", "mean"),
        AggregatorDependence=("AggregatorDependence", "mean"),
        SelfDeliveryShare=("SelfDeliveryShareCalc", "mean"),
        DeliveryRadiusKM=("DeliveryRadiusKM", "mean"),
        MonthlyOrders=("MonthlyOrders", "mean"),
        AOV=("AOV", "mean"),
        TotalNetProfit=("TotalNetProfit", "mean"),
    )
    st.dataframe(prof.round(3), use_container_width=True)
    metric = st.selectbox("Distribution of", ["GPI", "NetMargin", "AggregatorDependence", "CostBurden", "DeliveryRadiusKM", "MonthlyOrders", "TotalNetProfit"])
    st.plotly_chart(px.box(view, x="ClusterLabel", y=metric, color="ClusterLabel", height=460).update_layout(showlegend=False), use_container_width=True)

with tab_drivers:
    st.markdown("Share of variance in each lever that separates the archetypes (eta squared, higher means the lever matters more for classification).")
    lv = levers.reset_index()
    lv.columns = ["Lever", "EtaSquared"]
    st.plotly_chart(px.bar(lv.sort_values("EtaSquared"), x="EtaSquared", y="Lever", orientation="h", height=620), use_container_width=True)
    corr_cols = ["GPI", "NetMargin", "AggregatorDependence", "CostBurden", "CommissionDrag", "DeliveryCostRatio", "SelfDeliveryShareCalc", "DeliveryRadiusKM", "GrowthFactor", "AOV", "Scale"]
    corr = view[corr_cols].corr(method="spearman").round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, height=620), use_container_width=True)
