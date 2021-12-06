import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    FunctionTransformer,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
import pyarrow.parquet as pq
from utils.absorbance import absorbance


def preprocessor(data):
    """Preprocesses data.
    Only supports pandas dataframes.
    TODO: generalize to numpy/tensors?"""

    cols = data.columns

    target_col = [cols[0]]
    spectral_col = cols[1:13]
    quantitative_col = cols[np.r_[13:15, 16:22]]
    aspect_col = [cols[15]]
    categorical_col = cols[23:25]

    target_transformer = Pipeline(
            steps=[
                # ("reshape", FunctionTransformer(func=np.reshape, kw_args={
                # "newshape": (-1, 1)})),
                ("sqrt_scaler", FunctionTransformer(func=np.sqrt)),
                #("minmax_scaler", MinMaxScaler()),
            ]
    )

    # TODO: Implement aspect transformer with cos and sine col output
    aspect_transformer = Pipeline(
        steps=[
            ("deg_to_rad", FunctionTransformer(func=np.deg2rad)),
            ("cosine", FunctionTransformer(func=np.cos)),
        ]
    )

    quantitative_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("robust_scaler", RobustScaler()),
            ("minmax_scaler", MinMaxScaler()),
        ]
    )

    spectral_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("absorbance", FunctionTransformer(func=absorbance)),
            ("robust_scaler", RobustScaler()),
            ("minmax_scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("targets", target_transformer, target_col),
            ("spectral", spectral_transformer, spectral_col),
            ("quantitative", quantitative_transformer, quantitative_col),
            ("categorical", categorical_transformer, categorical_col),
        ],
        remainder="drop",
    )

    return preprocessor
