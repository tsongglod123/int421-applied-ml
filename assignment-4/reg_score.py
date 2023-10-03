from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Tuple

import pandas as pd
import numpy as np

RAND_STATE_VALUE = 0

def check_null(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    null_cols = df.columns[df.isnull().any()].to_list()
    null_df = pd.DataFrame(df.isnull().sum(), columns = ["null_count"])
    null_df["null_percentage"] = null_df["null_count"] / len(df)
    return null_df, null_cols

def convert_dtype(df: pd.DataFrame, cols: list, dtype: str) -> pd.DataFrame:
    df[cols] = df[cols].astype(dtype)
    return df

def fill_na(df: pd.DataFrame, num_features: list, cat_features: list) -> pd.DataFrame:
    _, cat_null_cols = check_null(df[cat_features])
    _, num_null_cols = check_null(df[num_features])
    
    df[cat_null_cols] = df[cat_null_cols].fillna("None")
    
    for col in num_null_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    
    return df

def get_reg_score(df: pd.DataFrame) -> float:
    to_str_cols = ["OverallQual", "OverallCond", "MSSubClass"]
    df_train = convert_dtype(df, to_str_cols, "str")

    numerical_features = df_train.select_dtypes(include = ["int64", "float64"]).columns.to_list()
    numerical_features.remove("SalePrice")
    categorical_features = df_train.select_dtypes(include = ["object"]).columns.to_list()

    df_train = fill_na(df_train, numerical_features, categorical_features)

    selected_features = ["OverallCond", "OverallQual", "GrLivArea", "MSZoning"]

    X = df_train[selected_features]
    y = df_train["SalePrice"]

    num_features = X.select_dtypes(include = ["int64", "float64"]).columns.to_list()
    cat_features = X.select_dtypes(include = ["object"]).columns.to_list()

    numerical_pipe = Pipeline(
        steps = [
            ("log_transform", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipe = Pipeline(
        steps = [
            ("label_encode", OneHotEncoder(handle_unknown = "ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numerical_pipe, num_features),
            ("cat", categorical_pipe, cat_features)
        ]
    )
    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = RAND_STATE_VALUE)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)
