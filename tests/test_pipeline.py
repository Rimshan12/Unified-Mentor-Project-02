import numpy as np
import pandas as pd
import pytest

from src.config import CLUSTER_FEATURES, DATA_PATH, GPI_WEIGHTS
from src.features import engineer_features, load_raw
from src.pipeline import run


@pytest.fixture(scope="module")
def raw():
    return load_raw(DATA_PATH)


@pytest.fixture(scope="module")
def feats(raw):
    return engineer_features(raw)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    scored, summary, metrics, *_ = run(out_path=out / "scored.csv", artifacts=out)
    return scored, summary, metrics


def test_raw_shape_and_ids(raw):
    assert raw.shape[0] == 1696
    assert raw["RestaurantID"].is_unique
    assert raw.isna().sum().sum() == 0


def test_order_totals_reconcile(raw):
    cols = ["InStoreOrders", "UberEatsOrders", "DoorDashOrders", "SelfDeliveryOrders"]
    assert (raw[cols].sum(axis=1) == raw["MonthlyOrders"]).all()


def test_profit_identity_instore(raw):
    expected = raw["InStoreRevenue"] * (1 - raw["COGSRate"] - raw["OPEXRate"])
    assert np.allclose(expected, raw["InStoreNetProfit"], atol=0.01)


def test_features_complete_and_finite(feats):
    assert set(CLUSTER_FEATURES).issubset(feats.columns)
    assert np.isfinite(feats[CLUSTER_FEATURES].to_numpy(dtype=float)).all()


def test_shares_are_valid(feats):
    total = (
        feats["InStoreShareCalc"]
        + feats["UberEatsShareCalc"]
        + feats["DoorDashShareCalc"]
        + feats["SelfDeliveryShareCalc"]
    )
    assert np.allclose(total, 1.0)
    assert feats["ChannelBalance"].between(0, 1).all()
    assert feats["AggregatorDependence"].between(0, 1).all()


def test_scale_kpi(feats):
    assert np.allclose(feats["Scale"], feats["MonthlyOrders"] * feats["GrowthFactor"])


def test_gpi_weights_sum_to_one():
    assert pytest.approx(sum(GPI_WEIGHTS.values())) == 1.0


def test_gpi_bounds(result):
    scored, _, _ = result
    assert scored["GPI"].between(0, 100).all()
    for c in GPI_WEIGHTS:
        assert scored[c].between(0, 100).all()


def test_every_restaurant_classified(result):
    scored, _, _ = result
    assert scored["ClusterLabel"].notna().all()
    assert scored["Recommendation"].isin(
        ["Optimize", "Rebalance channels", "Hold / Stabilize"]
    ).all()
    assert scored["ActionPlan"].str.len().gt(0).all()
    assert len(scored) == 1696


def test_cluster_count_and_labels_unique(result):
    scored, _, metrics = result
    assert scored["ClusterID"].nunique() == metrics["chosen_k"]
    assert scored.groupby("ClusterID")["ClusterLabel"].nunique().max() == 1
    assert scored["ClusterLabel"].nunique() == metrics["chosen_k"]


def test_cluster_min_size(result):
    scored, _, _ = result
    assert scored["ClusterID"].value_counts().min() >= 50


def test_high_gpi_cluster_has_positive_margin(result):
    scored, summary, _ = result
    top = summary.index[0]
    assert summary.loc[top, "NetMargin"] > 0


def test_optimize_has_higher_gpi_than_hold(result):
    scored, _, _ = result
    g = scored.groupby("Recommendation")["GPI"].mean()
    assert g["Optimize"] > g["Rebalance channels"]
    assert g["Optimize"] > g["Hold / Stabilize"]


def test_deterministic(tmp_path):
    a, *_ = run(out_path=tmp_path / "a.csv", artifacts=tmp_path)
    b, *_ = run(out_path=tmp_path / "b.csv", artifacts=tmp_path)
    pd.testing.assert_series_equal(a["ClusterLabel"], b["ClusterLabel"])
    assert np.allclose(a["GPI"], b["GPI"])
