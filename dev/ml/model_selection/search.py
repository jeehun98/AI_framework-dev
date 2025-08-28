import numpy as np
from itertools import product
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Callable

from .split import KFold, StratifiedKFold
from .validation import get_scorer
# 경량 스코어러(문자열/콜러블 모두 지원)

ParamGrid = Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]]

# -----------------------------
# 작은 유틸들
# -----------------------------
def _check_random_state(seed: Optional[int]):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("Invalid random_state")

def _clone_estimator(est):
    """
    매우 단순한 clone: 같은 클래스 인스턴스를 만들고 하이퍼파라미터 복사 시도.
    - 가능하면 get_params() 사용
    - 아니면 __dict__ shallow copy
    """
    cls = est.__class__
    try:
        if hasattr(est, "get_params"):
            params = est.get_params()
            return cls(**{k: v for k, v in params.items() if "__" not in k})
    except Exception:
        pass
    new = cls.__new__(cls)
    try:
        new.__dict__ = {**est.__dict__}
    except Exception:
        pass
    return new

def _set_params(est, **params):
    if hasattr(est, "set_params"):
        est.set_params(**params)
    else:
        # 파이프라인이 아니면 속성으로 직접 세팅
        for k, v in params.items():
            setattr(est, k, v)
    return est

def _fit_est(est, X, y):
    try:
        return est.fit(X, y)
    except TypeError:
        return est.fit(X)

def _predict_for_scoring(est, X, scoring_name: Optional[str]):
    """
    스코어러가 무엇인지에 따라 예측값/점수 추출.
    - 'roc_auc' → predict_proba/decision_function 점수
    - 그 외 → predict()
    """
    if isinstance(scoring_name, str) and scoring_name.lower() == "roc_auc":
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        if hasattr(est, "decision_function"):
            return est.decision_function(X)
    # fallback
    if hasattr(est, "predict"):
        return est.predict(X)
    if hasattr(est, "transform"):
        return est.transform(X)
    raise AttributeError("Estimator must implement predict()/predict_proba()/decision_function()/transform()")

def _infer_stratified(y: np.ndarray) -> bool:
    classes = np.unique(y)
    # 이산 레이블 + 비교적 적은 클래스면 분류로 판단
    is_discrete = (np.issubdtype(y.dtype, np.integer) or y.dtype.kind in "OUS")
    return bool(is_discrete and (len(classes) <= max(20, int(0.1 * len(y)))))

def _iter_param_grid(param_grid: ParamGrid) -> Iterator[Dict[str, Any]]:
    """GridSearch용: 카테시안 곱"""
    if isinstance(param_grid, dict):
        items = sorted(param_grid.items())
        keys = [k for k, _ in items]
        values = [v for _, v in items]
        for combo in product(*values):
            yield dict(zip(keys, combo))
    else:
        for grid in param_grid:
            for params in _iter_param_grid(grid):
                yield params

def _sample_from_space(space: Dict[str, Any], rng) -> Dict[str, Any]:
    """
    RandomizedSearch용: 값이
      - 시퀀스(list/tuple/ndarray)면 균등 샘플
      - rvs 메서드가 있으면 rvs() 호출(예: scipy.stats 배포)
      - callable이면 callable(rng) 호출
      - 단일 값이면 그대로
    """
    out = {}
    for k, v in space.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            out[k] = v[rng.randint(0, len(v))]
        elif hasattr(v, "rvs"):
            try:
                out[k] = v.rvs(random_state=rng)
            except TypeError:
                out[k] = v.rvs()
        elif callable(v):
            out[k] = v(rng)
        else:
            out[k] = v
    return out

# -----------------------------
# BaseSearchCV
# -----------------------------
class BaseSearchCV:
    def __init__(
        self,
        estimator,
        scoring: Union[str, Callable],
        n_splits: int,
        stratified: Optional[bool],
        shuffle: bool,
        random_state: Optional[int],
        refit: Union[bool, str, Callable] = True,
        return_train_score: bool = False,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_splits = n_splits
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.refit = refit
        self.return_train_score = return_train_score
        self.verbose = verbose

        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.scorer_ = None

    def _make_splits(self, X, y):
        if isinstance(self.n_splits, int):
            if self.stratified is None:
                use_strat = _infer_stratified(y)
            else:
                use_strat = self.stratified
            splitter = StratifiedKFold(self.n_splits, self.shuffle, self.random_state) if use_strat \
                       else KFold(self.n_splits, self.shuffle, self.random_state)
            return splitter.split(X, y) if use_strat else splitter.split(X)
        # 커스텀 splitter
        try:
            return self.n_splits.split(X, y)
        except TypeError:
            return self.n_splits.split(X)

    def _score_fold(self, est, X_tr, y_tr, X_va, y_va, scoring_name):
        # train
        train_score = None
        if self.return_train_score:
            y_tr_pred = _predict_for_scoring(est, X_tr, scoring_name)
            train_score = self.scorer_(y_tr, y_tr_pred)
        # valid
        y_va_pred = _predict_for_scoring(est, X_va, scoring_name)
        valid_score = self.scorer_(y_va, y_va_pred)
        return train_score, valid_score

    def _finalize_refit(self, X, y, best_params, scoring_name):
        if not self.refit:
            return None
        # refit=True → best_params로 전체 학습
        # refit='f1' 같은 문자열이면 그 스코어러로 베스트 선정 이미 됐으므로 동일
        est = _clone_estimator(self.estimator)
        _set_params(est, **best_params)
        _fit_est(est, X, y)
        return est

# -----------------------------
# GridSearchCV
# -----------------------------
class GridSearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator,
        param_grid: ParamGrid,
        scoring: Union[str, Callable] = "accuracy",
        cv: int = 5,
        stratified: Optional[bool] = None,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        return_train_score: bool = False,
        verbose: int = 0,
    ):
        super().__init__(estimator, scoring, cv, stratified, shuffle, random_state, refit, return_train_score, verbose)
        self.param_grid = param_grid

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        scoring_name = self.scoring if isinstance(self.scoring, str) else None
        self.scorer_ = get_scorer(self.scoring)

        splits = list(self._make_splits(X, y))
        results = {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
            "rank_test_score": [],
        }
        if self.return_train_score:
            results["mean_train_score"] = []
            results["std_train_score"] = []

        for params in _iter_param_grid(self.param_grid):
            if self.verbose:
                print(f"[GridSearch] params={params}")
            fold_scores = []
            fold_train_scores = []

            for tr_idx, va_idx in splits:
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                est = _clone_estimator(self.estimator)
                _set_params(est, **params)
                _fit_est(est, X_tr, y_tr)

                tr_s, va_s = self._score_fold(est, X_tr, y_tr, X_va, y_va, scoring_name)
                fold_scores.append(va_s)
                if self.return_train_score:
                    fold_train_scores.append(tr_s)

            mean = float(np.mean(fold_scores))
            std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

            results["params"].append(params)
            results["mean_test_score"].append(mean)
            results["std_test_score"].append(std)
            if self.return_train_score:
                tr_mean = float(np.mean(fold_train_scores))
                tr_std = float(np.std(fold_train_scores, ddof=1)) if len(fold_train_scores) > 1 else 0.0
                results["mean_train_score"].append(tr_mean)
                results["std_train_score"].append(tr_std)

        # 순위 계산 (내림차순: 높은 점수가 1위)
        ranks = np.argsort(np.argsort(-np.array(results["mean_test_score"]))) + 1
        results["rank_test_score"] = ranks.tolist()

        # 베스트 선택
        best_idx = int(np.argmax(results["mean_test_score"]))
        self.best_params_ = results["params"][best_idx]
        self.best_score_ = results["mean_test_score"][best_idx]
        self.best_estimator_ = self._finalize_refit(X, y, self.best_params_, scoring_name)

        self.cv_results_ = results
        return self

# -----------------------------
# RandomizedSearchCV
# -----------------------------
class RandomizedSearchCV(BaseSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions: Dict[str, Any],
        n_iter: int = 20,
        scoring: Union[str, Callable] = "accuracy",
        cv: int = 5,
        stratified: Optional[bool] = None,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        return_train_score: bool = False,
        verbose: int = 0,
    ):
        super().__init__(estimator, scoring, cv, stratified, shuffle, random_state, refit, return_train_score, verbose)
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        rng = _check_random_state(self.random_state)

        scoring_name = self.scoring if isinstance(self.scoring, str) else None
        self.scorer_ = get_scorer(self.scoring)

        splits = list(self._make_splits(X, y))
        results = {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
            "rank_test_score": [],
        }
        if self.return_train_score:
            results["mean_train_score"] = []
            results["std_train_score"] = []

        for it in range(self.n_iter):
            params = _sample_from_space(self.param_distributions, rng)
            if self.verbose:
                print(f"[RandomSearch] iter={it+1}/{self.n_iter} params={params}")

            fold_scores = []
            fold_train_scores = []

            for tr_idx, va_idx in splits:
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                est = _clone_estimator(self.estimator)
                _set_params(est, **params)
                _fit_est(est, X_tr, y_tr)

                tr_s, va_s = self._score_fold(est, X_tr, y_tr, X_va, y_va, scoring_name)
                fold_scores.append(va_s)
                if self.return_train_score:
                    fold_train_scores.append(tr_s)

            mean = float(np.mean(fold_scores))
            std = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

            results["params"].append(params)
            results["mean_test_score"].append(mean)
            results["std_test_score"].append(std)
            if self.return_train_score:
                tr_mean = float(np.mean(fold_train_scores))
                tr_std = float(np.std(fold_train_scores, ddof=1)) if len(fold_train_scores) > 1 else 0.0
                results["mean_train_score"].append(tr_mean)
                results["std_train_score"].append(tr_std)

        ranks = np.argsort(np.argsort(-np.array(results["mean_test_score"]))) + 1
        results["rank_test_score"] = ranks.tolist()

        best_idx = int(np.argmax(results["mean_test_score"]))
        self.best_params_ = results["params"][best_idx]
        self.best_score_ = results["mean_test_score"][best_idx]
        self.best_estimator_ = self._finalize_refit(X, y, self.best_params_, scoring_name)

        self.cv_results_ = results
        return self
