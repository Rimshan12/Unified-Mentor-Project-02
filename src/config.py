from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "skycity_restaurants.csv"
SCORED_PATH = ROOT / "data" / "restaurants_scored.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

RANDOM_STATE = 42

CHANNELS = ["InStore", "UberEats", "DoorDash", "SelfDelivery"]
ORDER_COLS = [f"{c}Orders" for c in CHANNELS]
REVENUE_COLS = [f"{c}Revenue" for c in CHANNELS]
PROFIT_COLS = [f"{c}NetProfit" for c in CHANNELS]

CLUSTER_FEATURES = [
    "Scale",
    "GrowthFactor",
    "AOV",
    "TotalRevenue",
    "NetMargin",
    "CostBurden",
    "CommissionRate",
    "CommissionDrag",
    "AggregatorDependence",
    "InStoreShareCalc",
    "SelfDeliveryShareCalc",
    "ChannelBalance",
    "DeliveryRadiusKM",
    "DeliveryCostPerOrder",
    "DeliveryCostRatio",
    "ExpansionHeadroom",
    "RevenueQuality",
    "AggregatorMargin",
    "SelfDeliveryMargin",
]

SKEWED_FEATURES = [
    "Scale",
    "TotalRevenue",
    "ExpansionHeadroom",
    "RevenueQuality",
    "DeliveryCostRatio",
]

CATEGORICAL_FEATURES = ["Segment", "CuisineType", "Subregion"]
CATEGORICAL_WEIGHT = 0.35

GPI_WEIGHTS = {
    "GrowthSignal": 0.30,
    "CostResilience": 0.30,
    "ChannelBalanceScore": 0.20,
    "LogisticsScalability": 0.20,
}

K_RANGE = range(3, 9)
K_PREFERRED = (4, 5, 6)
K_FINAL = 6
