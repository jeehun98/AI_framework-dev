import numpy as np
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

ArrayLike = Union[np.ndarray, Sequence]

# -----------------------------
# utils
# -----------------------------
def _check_random_state(seed: Optional[int]):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("Invalid random_state")

def _as_index_array(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)

def _shuffle_inplace(arr: np.ndarray, rng) -> None:
    rng.shuffle(arr)

def _safe_len(X) -> int:
    try:
        return len(X)
    except TypeError:
        # numpy scalar etc.
        return np.asarray(X).shape[0]

def _index_select(X: ArrayLike, idx: np.ndarray):
    """X가 array-like일 때 공통 인덱싱 (리스트, ndarray 모두 지원)"""
    if isinstance(X, np.ndarray):
        return X[idx]
    # list/tuple 등
    return [X[i] for i in idx]

# -----------------------------
# KFold
# -----------------------------
class KFold:
    """
    K-겹 교차검증 분할기.
    - n_splits: 폴드 수 (>=2)
    - shuffle: 폴드 전에 전체 인덱스 셔플
    - random_state: 재현성
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = _safe_len(X)
        idx = _as_index_array(n_samples)
        if self.shuffle:
            rng = _check_random_state(self.random_state)
            _shuffle_inplace(idx, rng)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            yield train_idx, val_idx
            current = stop

# -----------------------------
# StratifiedKFold
# -----------------------------
class StratifiedKFold:
    """
    레이블 분포를 폴드마다 최대한 보존하는 K-겹 분할기.
    - y가 필요함 (분류 레이블)
    - 각 클래스별 인덱스를 n_splits로 균등 분배
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: ArrayLike, y: ArrayLike) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        y = np.asarray(y)
        n_samples = _safe_len(y)
        if n_samples != _safe_len(X):
            raise ValueError("X and y have different lengths")

        rng = _check_random_state(self.random_state)
        # 클래스별 인덱스 목록
        classes, y_indices = np.unique(y, return_inverse=True)
        per_class_indices: List[List[int]] = [[] for _ in classes]
        for i, c_idx in enumerate(y_indices):
            per_class_indices[c_idx].append(i)
        # 셔플
        for lst in per_class_indices:
            lst_arr = np.array(lst, dtype=np.int64)
            if self.shuffle:
                _shuffle_inplace(lst_arr, rng)
            lst[:] = lst_arr.tolist()

        # 각 클래스 인덱스를 폴드별로 round-robin 배분
        folds = [[] for _ in range(self.n_splits)]
        for cls_idx_list in per_class_indices:
            for j, i in enumerate(cls_idx_list):
                folds[j % self.n_splits].append(i)

        # 생성
        all_idx = set(range(n_samples))
        for f in range(self.n_splits):
            val_idx = np.array(sorted(folds[f]), dtype=np.int64)
            train_idx = np.array(sorted(list(all_idx.difference(folds[f]))), dtype=np.int64)
            yield train_idx, val_idx

# -----------------------------
# train_test_split
# -----------------------------
def train_test_split(
    *arrays: ArrayLike,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
):
    """
    배열들을 동일 인덱스 기준으로 학습/테스트로 분할.
    - test_size: (0,1) 비율 또는 정수 샘플 수
    - train_size: 지정 시 test_size와 합이 길이와 일치해야 함
    - stratify: y 또는 레이블 배열을 주면 층화 분할
    반환: 각 배열의 (X_train, X_test) 쌍을 순서대로
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required")
    n_samples = _safe_len(arrays[0])
    for a in arrays[1:]:
        if _safe_len(a) != n_samples:
            raise ValueError("All input arrays must have the same length")

    # 크기 결정
    if test_size is None and train_size is None:
        test_size = 0.25
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size as float must be in (0,1)")
        n_test = int(np.floor(n_samples * test_size))
    elif isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    elif test_size is None and train_size is not None:
        if isinstance(train_size, float):
            if not 0.0 < train_size < 1.0:
                raise ValueError("train_size as float must be in (0,1)")
            n_train = int(np.floor(n_samples * train_size))
        else:
            n_train = int(train_size)
        n_test = n_samples - n_train
    else:
        n_test = int(np.floor(n_samples * 0.25))

    if not 1 <= n_test <= n_samples - 1:
        raise ValueError("test set size must be between 1 and n_samples-1")

    rng = _check_random_state(random_state)
    indices = _as_index_array(n_samples)

    if stratify is None:
        # 일반 분할
        if shuffle:
            _shuffle_inplace(indices, rng)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        # 층화 분할
        y = np.asarray(stratify)
        if y.shape[0] != n_samples:
            raise ValueError("stratify array has different length from inputs")
        classes, y_indices = np.unique(y, return_inverse=True)

        # 클래스별 인덱스 수집 + 셔플
        class_to_idx = {}
        for c in range(len(classes)):
            cls_idx = np.where(y_indices == c)[0]
            if shuffle:
                rng.shuffle(cls_idx)
            class_to_idx[c] = cls_idx

        # 각 클래스별 테스트 샘플 수 비율로 배분 (총합 보정)
        test_counts = []
        residuals = []
        for c, idxs in class_to_idx.items():
            exact = len(idxs) * (n_test / n_samples)
            cnt = int(np.floor(exact))
            test_counts.append(cnt)
            residuals.append((exact - cnt, c))
        # 남은 샘플은 큰 잔여(residual) 순으로 1개씩 추가
        missing = n_test - sum(test_counts)
        for _, c in sorted(residuals, key=lambda x: x[0], reverse=True)[:missing]:
            test_counts[c] += 1

        # 실제 인덱스 선택
        test_idx_list = []
        train_idx_list = []
        for c, idxs in class_to_idx.items():
            t = test_counts[c]
            t_idx = idxs[:t]
            tr_idx = idxs[t:]
            test_idx_list.append(t_idx)
            train_idx_list.append(tr_idx)

        test_idx = np.concatenate(test_idx_list).astype(np.int64)
        train_idx = np.concatenate(train_idx_list).astype(np.int64)

        # 섞여있지 않다면 순서 고정 (가독성용)
        test_idx = test_idx if shuffle else np.sort(test_idx)
        train_idx = train_idx if shuffle else np.sort(train_idx)

    # 출력 구성
    out = []
    for arr in arrays:
        A = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
        out.extend((_index_select(A, train_idx), _index_select(A, test_idx)))
    return tuple(out)
