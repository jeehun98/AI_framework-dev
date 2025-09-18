# ge2_min_experiments.py

# 디버그용 run_case를 가져온다.
# 파일명이 다르면 아래 import를 네 파일명으로 변경하세요.
from ge2_check_backward_debug import run_case
import graph_executor_v2 as ge2


SEP = "=" * 72

def test_case2_save_preact_off():
    """
    [실험1] 케이스2에서 Z 저장 경로 배제 (save_preact=False)
    - 의도: FWD가 pre 대신 post를 Z에 저장하거나, BWD의 ldZ 해석 문제가 있는지 확인
    """
    print(SEP)
    print("[TEST 1] Case2: save_preact=False (CPU pre로 대체)")
    run_case(M=64, N=64, K=32,
             act_kind=ge2.ActKind.ReLU,
             bias_kind=ge2.BiasKind.Scalar,
             use_C=False, save_preact=True, alpha=1.0)

def test_case2_scalar_to_pern():
    """
    [실험2] 케이스2에서 Scalar → PerN 치환
    - 의도: Scalar 전용 reduce 경로가 문제인지 확인
    """
    print(SEP)
    print("[TEST 2] Case2: bias Scalar → PerN")
    run_case(M=64, N=64, K=32,
             act_kind=ge2.ActKind.ReLU,
             bias_kind=ge2.BiasKind.PerN,
             use_C=False, save_preact=True)

def test_case2_no_bias():
    """
    [실험3] 케이스2에서 bias 완전 OFF
    - 의도: bias가 켜질 때 타는 BWD 코드패스가 dA/dB까지 오염시키는지 확인
    """
    print(SEP)
    print("[TEST 3] Case2: bias OFF (BiasKind.None)")
    run_case(M=64, N=64, K=32,
             act_kind=ge2.ActKind.ReLU,
             bias_kind=getattr(ge2.BiasKind, "None"),
             use_C=False, save_preact=True)

def test_case3_branch():
    """
    [실험4] 케이스3 원인 분기
      (a) PerM → PerN 교체
      (b) Leaky → ReLU 교체
      (c) Leaky slope=0(=ReLU 동치)
    - 의도: PerM 축/브로드캐스트 vs Leaky enum/슬로프 문제를 분리
    """
    print(SEP)
    print("[TEST 4a] Case3: PerM → PerN")
    run_case(M=96, N=80, K=40,
             act_kind=ge2.ActKind.LeakyReLU,
             bias_kind=ge2.BiasKind.PerN,
             use_C=True, save_preact=True, leaky=0.02)

    print(SEP)
    print("[TEST 4b] Case3: Leaky → ReLU (PerM 유지)")
    run_case(M=96, N=80, K=40,
             act_kind=ge2.ActKind.ReLU,
             bias_kind=ge2.BiasKind.PerM,
             use_C=True, save_preact=True)

    print(SEP)
    print("[TEST 4c] Case3: Leaky slope=0.0 (PerM 유지, ReLU와 동일 동작)")
    run_case(M=96, N=80, K=40,
             act_kind=ge2.ActKind.LeakyReLU,
             bias_kind=ge2.BiasKind.PerM,
             use_C=True, save_preact=True, leaky=0.0)

if __name__ == "__main__":
    # ===== 최소 실험 4개 =====
    test_case2_save_preact_off()
    test_case2_scalar_to_pern()
    test_case2_no_bias()
    test_case3_branch()
