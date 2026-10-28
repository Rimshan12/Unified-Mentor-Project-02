import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_WEIGHT,
    CLUSTER_FEATURES,
    SKEWED_FEATURES,
)


def build_matrix(df):
    num = df[CLUSTER_FEATURES].astype(float).copy()
    num = num.fillna(num.median())

    skew_cols = [c for c in SKEWED_FEATURES if c in num.columns]
    if skew_cols:
        shifted = num[skew_cols] - num[skew_cols].min() + 1e-6
        num[skew_cols] = np.log1p(shifted)

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    num = pd.DataFrame(pt.fit_transform(num), columns=num.columns, index=num.index)
    scaled = StandardScaler().fit_transform(num)
    num_df = pd.DataFrame(scaled, columns=CLUSTER_FEATURES, index=df.index)

    cat = pd.get_dummies(df[CATEGORICAL_FEATURES], dtype=float)
    cat = cat * CATEGORICAL_WEIGHT

    X = pd.concat([num_df, cat], axis=1)
    return X, num_df, list(cat.columns)
