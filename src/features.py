import numpy as np
import pandas as pd

from .config import ORDER_COLS, PROFIT_COLS, REVENUE_COLS

REQUIRED_COLUMNS = (
    ["RestaurantID", "RestaurantName", "CuisineType", "Segment", "Subregion"]
    + ["GrowthFactor", "AOV", "MonthlyOrders", "COGSRate", "OPEXRate"]
    + ["CommissionRate", "DeliveryRadiusKM", "DeliveryCostPerOrder"]
    + ["SD_DeliveryTotalCost"]
    + ORDER_COLS
    + REVENUE_COLS
    + PROFIT_COLS
)


def load_raw(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def _safe_div(a, b):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    return np.asarray(a) / b


def _normalized_entropy(shares):
    shares = np.clip(shares, 1e-12, 1.0)
    ent = -(shares * np.log(shares)).sum(axis=1)
    return ent / np.log(shares.shape[1])


def engineer_features(df):
    d = df.copy()
    orders = d[ORDER_COLS].to_numpy(dtype=float)
    revenue = d[REVENUE_COLS].to_numpy(dtype=float)
    profit = d[PROFIT_COLS].to_numpy(dtype=float)

    d["TotalRevenue"] = revenue.sum(axis=1)
    d["TotalNetProfit"] = profit.sum(axis=1)
    d["NetMargin"] = _safe_div(d["TotalNetProfit"], d["TotalRevenue"])

    d["Scale"] = d["MonthlyOrders"] * d["GrowthFactor"]
    d["CostBurden"] = d["COGSRate"] + d["OPEXRate"]
    d["CostDiscipline"] = 1.0 - d["CostBurden"]

    order_share = orders / orders.sum(axis=1, keepdims=True)
    d["InStoreShareCalc"] = order_share[:, 0]
    d["UberEatsShareCalc"] = order_share[:, 1]
    d["DoorDashShareCalc"] = order_share[:, 2]
    d["SelfDeliveryShareCalc"] = order_share[:, 3]
    d["AggregatorDependence"] = order_share[:, 1] + order_share[:, 2]
    d["ChannelBalance"] = _normalized_entropy(order_share)

    agg_rev = d["UberEatsRevenue"] + d["DoorDashRevenue"]
    agg_profit = d["UberEatsNetProfit"] + d["DoorDashNetProfit"]
    d["AggregatorMargin"] = _safe_div(agg_profit, agg_rev)
    d["SelfDeliveryMargin"] = _safe_div(d["SelfDeliveryNetProfit"], d["SelfDeliveryRevenue"])
    d["InStoreMargin"] = _safe_div(d["InStoreNetProfit"], d["InStoreRevenue"])
    d["CommissionDrag"] = _safe_div(agg_rev * d["CommissionRate"], d["TotalRevenue"])

    d["DeliveryCostRatio"] = _safe_div(d["SD_DeliveryTotalCost"], d["SelfDeliveryRevenue"])

    delivery_orders = d["UberEatsOrders"] + d["DoorDashOrders"] + d["SelfDeliveryOrders"]
    d["DeliveryOrders"] = delivery_orders
    d["ExpansionHeadroom"] = d["DeliveryRadiusKM"] / (delivery_orders / 100.0)

    channel_margins = np.column_stack(
        [d["InStoreMargin"], _safe_div(d["UberEatsNetProfit"], d["UberEatsRevenue"]),
         _safe_div(d["DoorDashNetProfit"], d["DoorDashRevenue"]), d["SelfDeliveryMargin"]]
    )
    d["MarginConsistency"] = 1.0 / (1.0 + np.nanstd(channel_margins, axis=1))
    d["RevenueQuality"] = d["AOV"] * d["NetMargin"] * d["MarginConsistency"]

    return d.replace([np.inf, -np.inf], np.nan)
