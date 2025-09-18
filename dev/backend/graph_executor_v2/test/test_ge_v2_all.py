

import graph_executor_v2 as ge2

def test_smoke_build_only():
    # This is a build/link smoke test; actual device memory test is out-of-scope here.
    p = ge2.GemmBiasActParams()
    p.M = 4; p.N = 4; p.K = 4
    p.has_bias = 0; p.act = 0
    # We can't run without real device pointers; ensure the binding is importable and struct works.
    assert p.M == 4 and p.N == 4 and p.K == 4
