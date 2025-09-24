import os, sys, numpy as np

# === Import path & DLL 경로 설정 ===
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

from graph_executor_v2 import _core as ge


def report(name, y, ref, atol=1e-6):
    ok = np.allclose(y, ref, atol=atol)
    print(f"{name} close: {ok}")
    if not ok:
        diff = np.abs(y - ref)
        print("  y.shape:", y.shape, "ref.shape:", ref.shape)
        print("  max|diff|:", diff.max(), "argmax:", np.unravel_index(diff.argmax(), diff.shape))
        print("  y sample:\n", y)
        print("  ref sample:\n", ref)
    return ok


def test_reduce_sum():
    print("=== test_reduce_sum ===")
    X = np.random.randn(2, 3, 4).astype(np.float32)

    y = ge.reduce_sum(X, axes=[-1], keepdim=False)
    ref = X.sum(axis=-1)
    report("sum last", y, ref)

    y = ge.reduce_sum(X, axes=[0,2], keepdim=True)
    ref = X.sum(axis=(0,2), keepdims=True)
    report("sum [0,2] keepdim", y, ref)

    y = ge.reduce_sum(X, axes=None, keepdim=False)
    ref = X.sum()
    report("sum all", y, ref)


def test_reduce_mean():
    print("=== test_reduce_mean ===")
    X = np.random.randn(2, 3, 4).astype(np.float32)

    # mean over [0,2] keepdim
    y = ge.reduce_mean(X, axes=[0, 2], keepdim=True)
    ref = X.mean(axis=(0, 2), keepdims=True)
    report("mean [0,2] keepdim", y, ref)

    # 분해 검사: mean == sum / prod(reduced)
    y_sum = ge.reduce_sum(X, axes=[0,2], keepdim=True)
    reduce_elems = X.shape[0] * X.shape[2]
    ref2 = y_sum / reduce_elems
    report("mean via sum/prod", y, ref2)

    y = ge.reduce_mean(X, axes=None, keepdim=False)
    ref = X.mean()
    report("mean all", y, ref)


def test_reduce_max_min():
    print("=== test_reduce_max/min ===")
    X = np.random.randn(2, 3, 4).astype(np.float32)

    y = ge.reduce_max(X, axes=None, keepdim=False)
    ref = X.max()
    report("max all", y, ref)

    y = ge.reduce_min(X, axes=[1], keepdim=False)
    ref = X.min(axis=1)
    report("min axis=1", y, ref)

    y = ge.reduce_min(X, axes=[-1], keepdim=True)
    ref = X.min(axis=-1, keepdims=True)
    report("min last keepdim", y, ref)


def main():
    np.random.seed(0)
    test_reduce_sum()
    test_reduce_mean()
    test_reduce_max_min()


if __name__ == "__main__":
    main()
