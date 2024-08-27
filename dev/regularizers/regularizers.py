import math
import numpy as np

# 계산 결과값을 weight 에 더해야 한다!!!!

class Regularizer:
    def __call__(self, x):
        # 자식 클래스에서 이 부분의 구현을 해야 함
        return 0

class L1(Regularizer):
    def __init__(self, l1=0.01):
        self.l1 = l1

    def __call__(self, weights):
        """
        L1 정규화 연산을 수행합니다.
        
        Args:
            weights (numpy.ndarray): 정규화할 가중치 행렬.
            l1_factor (float): L1 정규화 강도.

        Returns:
            float: L1 정규화 손실 값.
        """
        return self.l1 * np.sum(np.abs(weights))

class L2(Regularizer):
    def __init__(self, l2=0.01):
        self.l2 = l2

    def __call__(self, weights):
        """
        L2 정규화 연산을 수행합니다.
        
        Args:
            weights (numpy.ndarray): 정규화할 가중치 행렬.
            l2_factor (float): L2 정규화 강도.

        Returns:
            float: L2 정규화 손실 값.
        """
        return self.l2 * np.sum(np.square(weights))
    
class L1L2(Regularizer):
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def l1_l2_regularization(self, weights):
        """
        L1과 L2 정규화 연산을 동시에 수행합니다.
        
        Args:
            weights (numpy.ndarray): 정규화할 가중치 행렬.
            l1_factor (float): L1 정규화 강도.
            l2_factor (float): L2 정규화 강도.

        Returns:
            float: L1 및 L2 정규화 손실 값의 합.
        """
        l1_loss = self.l1 * np.sum(np.abs(weights))
        l2_loss = self.l2 * np.sum(np.square(weights))
        return l1_loss + l2_loss
    
class Orthogonal(Regularizer):
    """
    직교 정규화 연산을 수행합니다.
    
    Args:
        weights (numpy.ndarray): 정규화할 가중치 행렬.
        factor (float): 정규화 강도.
        mode (str): "rows" 또는 "columns", 정규화를 적용할 축을 결정합니다.

    Returns:
        float: 직교 정규화 손실 값.
    """
    def __init__(self, factor=0.01, mode="rows"):
        self.factor = factor
        self.mode = mode

    def __call__(self, weights):
        if self.mode == "rows":
            normalized = weights / np.linalg.norm(weights, axis=1, keepdims=True)
            product = np.dot(normalized, normalized.T)
        elif self.mode == "columns":
            normalized = weights / np.linalg.norm(weights, axis=0, keepdims=True)
            product = np.dot(normalized.T, normalized)
        else:
            raise ValueError("Invalid mode. Use 'rows' or 'columns'.")

        size = product.shape[0]
        product_no_diagonal = product - np.eye(size)
        num_pairs = size * (size - 1) / 2.0
        return self.factor * 0.5 * np.sum(np.abs(product_no_diagonal)) / num_pairs