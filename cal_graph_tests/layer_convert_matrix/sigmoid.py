import numpy as np
import pandas as pd

# ---------------------------
# Sigmoid(x) = 1 / (1 + exp(-x))
# exp(-x) 다항 근사 (6차): 1 - x + x^2/2 - x^3/6 + x^4/24 - x^5/120 + x^6/720
# ---------------------------

x = 0.3

# A_neg = [1, -x, x^2, -x^3, x^4, -x^5, x^6]
A_neg = np.array([[1.0, -x, x**2, -x**3, x**4, -x**5, x**6]])

# 계수 벡터: [1, 1, 1/2, 1/6, 1/24, 1/120, 1/720]
B_exp = np.array([[1.0], [1.0], [1/2], [1/6], [1/24], [1/120], [1/720]])

# exp(-x) 근사값
exp_neg_approx = A_neg @ B_exp

# 1 + exp(-x)
denom = 1.0 + float(exp_neg_approx)

# sigmoid 근사
sigmoid_approx = 1.0 / denom

# 실제 sigmoid 값
sigmoid_true = 1.0 / (1.0 + np.exp(-x))

# 결과 정리
df_sigmoid_compare = pd.DataFrame({
    "Expression": ["sigmoid_approx (6th-order)", "sigmoid_true (np version)"],
    "Value": [sigmoid_approx, sigmoid_true]
})

print(df_sigmoid_compare)