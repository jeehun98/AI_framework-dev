import numpy as np

from ml.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    SimpleImputer, OneHotEncoder, PolynomialFeatures,
    Pipeline, ColumnTransformer, make_column_selector
)

def test_all():
    # 혼합 데이터
    X = np.array([
        [25.0, 3000.0, "red",   "S"],
        [None, 5200.0, "blue",  None],
        [43.0, None,   "red",   "M"],
        [36.0, 4100.0, "green", "L"],
    ], dtype=object)
    y = np.array([0, 1, 0, 1])

    num_sel = make_column_selector(dtype_include=np.number)
    cat_sel = make_column_selector(dtype_include=np.object_)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_sel),
        ("cat", cat_pipe, cat_sel),
    ], remainder="drop")

    Xt = pre.fit_transform(X, y)
    print("Transformed shape:", Xt.shape)

if __name__ == "__main__":
    test_all()
