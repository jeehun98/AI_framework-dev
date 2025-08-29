import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from backend import CPUBackend
from ml.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

be = CPUBackend()

def test_ridge_small():
    X = np.array([[1.,2.],[2.,0.],[3.,1.],[4.,3.],[5.,4.]])
    y = np.array([3.2, 2.1, 3.0, 4.9, 6.2])
    model = Ridge(alpha=1.0, backend=be).fit(X, y)
    yhat = model.predict(X)
    assert yhat.shape == y.shape

def test_logistic_ovr():
    X = np.array([
        [25.0,3000.0],[30.0,4200.0],[40.0,5000.0],[35.0,4100.0],
        [28.0,4700.0],[33.0,3900.0],[41.0,4500.0],[27.0,5100.0],
    ])
    y = np.array([0,1,0,1,1,0,1,0])
    clf = LogisticRegression(lr=0.2, max_iter=500, backend=be).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 2)

if __name__ == "__main__":
    test_ridge_small()
    test_logistic_ovr()
    print("CPU linear models OK")
