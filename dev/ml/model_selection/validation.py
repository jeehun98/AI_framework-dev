import numpy as np
from typing import Callable, Optional, Sequence, Union, List, Tuple

from .split import KFold, StratifiedKFold
from ..metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score,
)

Scorer = Callable[[np.ndarray, np.ndarray], float]

# -----------------------------
# Scorer factory
# -----------------------------
def get_scorer(name: Union[str, Scorer], task: Optional[str]=None, average: str="binary") -> Scorer:
    """
    name: 'accuracy' | 'precision' | 'recall' | 'f1' | 'roc_auc' |
          'neg_mean_squared_error' | 'r2' | callable
    task 힌트: 'classification' | 'regression' (없어도 동작)
    """
    if callable(name):
        return name

    nm = name.lower()
    if nm == "accuracy":
        return lambda y_true, y_pred, **kw: accuracy_score(y_true, y_pred)
    if nm == "precision":
        return lambda y_true, y_pred, **kw: precision_score(y_true, y_pred, average=average)
    if nm == "recall":
        return lambda y_true, y_pred, **kw: recall_score(y_true, y_pred, average=average)
    if nm == "f1":
        return lambda y_true, y_pred, **kw: f1_score(y_true, y_pred, average=average)
    if nm == "roc_auc":
        # y_pred는 확률/점수 필요 → 관례적으로 proba[:,1] 또는 decision_function 사용
        def _auc(y_true, y_pred_or_score, **kw):
            return roc_auc_score(y_true, y_pred_or_score)
        return _auc
    if nm in ("neg_mean_squared_error", "neg_mse"):
        return lambda y_true, y_pred, **kw: -mean_squared_error(y_true, y_pred)
    if nm == "r2":
        return lambda y_true, y_pred, **kw: r2_score(y_true, y_pred)

    raise ValueError(f"Unknown scorer: {name}")

# -----------------------------
# cross_val_score
# -----------------------------
def cross_val_score(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: Optional[int]=5,
    scoring: Union[str, Scorer]="accuracy",
    stratified: Optional[bool]=None,
    shuffle: bool=True,
    random_state: Optional[int]=None,
    return_predictions: bool=False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    간단한 교차검증 스코어링.
    - cv: int -> (Stratified)KFold n_splits
    - scoring: 문자열 또는 callable(y_true, y_pred/score)->float
      * 'roc_auc'는 estimator가 predict_proba() 또는 decision_function()을 지원해야 함
    - stratified: None이면 y의 고유 클래스 수가 적으면 분류로 간주하여 StratifiedKFold 사용
    - return_predictions: True면 (scores, y_pred_cv) 반환
    """
    X = np.asarray(X, dtype=object)  # 혼합 타입 허용(전처리 파이프라인 앞단 가능)
    y = np.asarray(y)

    # CV splitter 결정
    if isinstance(cv, int):
        if stratified is None:
            # 분류로 추정: 클래스 개수 적고 데이터 타입이 이산적이면
            classes = np.unique(y)
            is_class = (np.issubdtype(y.dtype, np.integer) or y.dtype.kind in "OUS") and (len(classes) <= max(20, int(0.1*len(y))))
            stratified = is_class
        splitter = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state) if stratified \
                   else KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        splits = splitter.split(X, y) if stratified else splitter.split(X)
    else:
        # 사용자가 직접 splitter 제공한 경우
        splitter = cv
        try:
            splits = splitter.split(X, y)
        except TypeError:
            splits = splitter.split(X)

    # 스코어러 준비
    # AUC면 점수 필요 → predict_proba / decision_function 선호
    use_scores = (isinstance(scoring, str) and scoring.lower()=="roc_auc")
    scorer = get_scorer(scoring)

    scores: List[float] = []
    preds_all = np.empty_like(y, dtype=float if use_scores else y.dtype)

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        est = _clone_like(estimator)
        _fit_est(est, X_tr, y_tr)

        if use_scores:
            y_va_pred = _get_decision_score(est, X_va, y_va)
        else:
            y_va_pred = _predict_general(est, X_va)

        preds_all[va_idx] = y_va_pred
        score = scorer(y_va, y_va_pred)
        scores.append(float(score))

    scores = np.array(scores, dtype=float)
    return (scores, preds_all) if return_predictions else scores

# -----------------------------
# small helpers
# -----------------------------
def _clone_like(est):
    # 매우 단순한 클론: 동일 클래스 재생성 + __dict__ 하이퍼파라미터 복사 시도
    cls = est.__class__
    new = cls.__new__(cls)
    try:
        new.__dict__ = {**est.__dict__}  # shallow copy (하이퍼파라미터만 복사 기대)
    except Exception:
        pass
    # __init__ 호출이 필요한 모델이라면 직접 재생성하는 방식을 권장
    try:
        if hasattr(est, "get_params"):
            params = est.get_params()
            return cls(**{k:v for k,v in params.items() if "__" not in k})
    except Exception:
        pass
    return new if hasattr(new, "fit") else cls()

def _fit_est(est, X, y):
    try:
        return est.fit(X, y)
    except TypeError:
        return est.fit(X)

def _predict_general(est, X):
    if hasattr(est, "predict"):
        return est.predict(X)
    # 마지막 단계가 변환기일 수 있는 파이프라인 등 특수 상황 회피용
    if hasattr(est, "transform"):
        return est.transform(X)
    raise AttributeError("Estimator must implement predict() or transform()")

def _get_decision_score(est, X, y):
    # AUC용 점수 추출: predict_proba[:,1] > decision_function > predict (fallback)
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    # fallback: 예측 라벨을 점수로 쓰면 AUC 정보력 떨어지지만 마지막 수단
    pred = est.predict(X)
    return pred.astype(float)
