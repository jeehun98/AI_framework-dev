#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>
#include <cuda_runtime_api.h>

namespace gev2 {

struct AllocStats {
    size_t total_reserved = 0;  // cudaMalloc 총합
    size_t peak_in_use    = 0;  // 동시에 사용된 최대 바이트
    size_t curr_in_use    = 0;  // 현재 사용 중
    int    slabs          = 0;  // 슬랩 개수
};

class CaptureSafeArena {
public:
    static CaptureSafeArena& instance();

    // 캡처 "전에" 필요한 만큼 예약(슬랩 추가). 부족하면 cudaMalloc로 슬랩을 늘린다.
    void reserve_bytes(size_t bytes);

    // 임시 버퍼 대여/반납 (LIFO 우선)
    // 반환값: 디바이스 포인터 (uint64로 Python에 넘기기 편하게)
    uint64_t alloc_temp(size_t nbytes, size_t align, cudaStream_t stream);
    void     free_temp(uint64_t ptr, cudaStream_t stream);

    // 풀 재설정(슬랩 유지, 오프셋/스택만 리셋)
    void reset_pool();

    // 전체 해제(옵션): 디스트럭터에서 호출되므로 명시 호출은 비권장
    void destroy_all();

    AllocStats stats() const;

private:
    CaptureSafeArena() = default;
    ~CaptureSafeArena();

    struct Slab {
        void*  base = nullptr;
        size_t bytes = 0;
        size_t bump  = 0;    // bump pointer
    };

    // 간단 LIFO 스택 노드(필요 시 free-list로 고도화)
    struct StackNode { uint64_t ptr; size_t size; };

    // 내부 유틸
    bool is_stream_capturing(cudaStream_t s) const;
    static size_t align_up(size_t x, size_t a) { return (x + (a-1)) & ~(a-1); }

private:
    mutable std::mutex m_;
    std::vector<Slab> slabs_;
    std::vector<StackNode> lifo_;  // 최근 반납 블록을 우선 재사용
    size_t total_reserved_ = 0;
    size_t peak_in_use_    = 0;
    size_t curr_in_use_    = 0;
};

} // namespace gev2
