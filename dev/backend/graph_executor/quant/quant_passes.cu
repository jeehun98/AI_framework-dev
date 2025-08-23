#include <algorithm>
#include "quant_types.cuh"
#include "observers.cuh"
#include "quant_kernels.cuh"

namespace quant {

// 공개 API 1) 옵저버 on/off
void insert_observers(){ enable_observers(true); }
void clear_observers(){ cache().act_minmax.clear(); enable_observers(false); }

// 공개 API 2) 캘리브레이션 종료 → scale/zp 고정
void freeze_calibration(){ freeze_act_qparams(); }

// 공개 API 3) 가중치 INT8 변환 (대칭 per-channel)
//  - W: float[OC,K], weight_id: 고유 문자열
void convert_weight_to_int8(const std::string& weight_id, const float* dW, int OC, int K, WeightInt8& out){
    // per-channel amax → scale
    std::vector<float> h_scales(OC, 0.f);
    std::vector<float> h_amax(OC, 0.f);

    // 블록별 absolute max를 CPU에서 간단히: 각 row를 한번에 memcpy 후 reduce (메모리 고려해 스트라이드 접근)
    // 간단화를 위해 여기선 장치에서 절대값 최대를 구하지 않고 host로 한 줄씩 복사 (실서비스에서는 전용 커널 권장)
    std::vector<float> hrow(K);
    for (int oc=0; oc<OC; ++oc){
        CUDA_CHECK(cudaMemcpy(hrow.data(), dW + oc*K, K*sizeof(float), cudaMemcpyDeviceToHost));
        float amax = 0.f;
        for (int k=0;k<K;++k) amax = std::max(amax, fabsf(hrow[k]));
        if (amax < 1e-8f) amax = 1e-8f;
        h_amax[oc] = amax;
        h_scales[oc] = amax / 127.f;
    }

    // 디바이스로 scale 전송
    float* d_scales=nullptr;
    CUDA_CHECK(cudaMalloc(&d_scales, OC*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), OC*sizeof(float), cudaMemcpyHostToDevice));

    // Wq(row-major) 할당/양자화
    int8_t* d_Wq_row=nullptr; CUDA_CHECK(cudaMalloc(&d_Wq_row, OC*K*sizeof(int8_t)));
    k_quantize_w_s8_per_channel<<<OC, 256>>>(dW, OC, K, d_scales, d_Wq_row);
    CUDA_CHECK(cudaPeekAtLastError());

    // col-major 패킹
    int8_t* d_Wq_col=nullptr; CUDA_CHECK(cudaMalloc(&d_Wq_col, OC*K*sizeof(int8_t))); // same size
    dim3 blk(32, 8); dim3 grd((K+blk.x-1)/blk.x, (OC+blk.y-1)/blk.y);
    k_pack_row_to_col<<<grd, blk>>>(d_Wq_row, OC, K, d_Wq_col);
    CUDA_CHECK(cudaPeekAtLastError());

    // out 채우기
    out.row_major = d_Wq_row;
    out.col_major = d_Wq_col;
    out.per_channel_scales = d_scales; // 소유권 유지
    out.OC = OC; out.K = K;
}

void register_weight_q(const std::string& weight_id, const float* dW, int OC, int K){
    WeightInt8 wq;
    convert_weight_to_int8(weight_id, dW, OC, K, wq);
    cache().weights_q[weight_id] = wq;
}

// 공개 API 4) 런타임 스위치
void enable_quant(bool on){ runtime().quant_enabled = on; }

// 헬퍼: 활성 quantize 런칭
void quantize_activation(const std::string& tensor_id, const float* X, int n, int8_t* Xq){
    auto it = cache().act_qparams.find(tensor_id);
    if (it == cache().act_qparams.end()){
        // 캘리브 안된 경우 안전하게 관찰-동결 후 계산
        MinMax mm; mm.min=-1.f; mm.max=1.f; mm.initialized=true;
        cache().act_qparams[tensor_id] = calc_qparams_activation(mm);
        it = cache().act_qparams.find(tensor_id);
    }
    const QuantParams qp = it->second;
    const int threads=256, blocks=std::min((n+threads-1)/threads, 65535);
    k_quantize_act_s8<<<blocks, threads>>>(X, n, qp, Xq);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace quant
