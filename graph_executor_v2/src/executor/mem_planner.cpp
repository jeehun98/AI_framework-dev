#include "mem_planner.hpp"
#include <stdexcept>
#include <algorithm>

using namespace gev2;

CaptureSafeArena& CaptureSafeArena::instance() {
    static CaptureSafeArena g;
    return g;
}

CaptureSafeArena::~CaptureSafeArena() { destroy_all(); }

bool CaptureSafeArena::is_stream_capturing(cudaStream_t s) const {
    cudaStreamCaptureStatus status;
    auto st = cudaStreamIsCapturing(s, &status);
    if (st == cudaErrorStreamCaptureUnsupported) return false;
    if (st != cudaSuccess) return false;
    return status != cudaStreamCaptureStatusNone;
}

void CaptureSafeArena::reserve_bytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(m_);
    size_t have = 0;
    for (auto& sl : slabs_) have += sl.bytes;
    if (have >= bytes) return;

    size_t need = bytes - have;
    // 한 번에 크게 땡겨오되(슬랩), 256바이트 정렬
    size_t slab_bytes = align_up(need, 256);

    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, slab_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("CaptureSafeArena::reserve_bytes cudaMalloc failed");
    }
    Slab s; s.base = p; s.bytes = slab_bytes; s.bump = 0;
    slabs_.push_back(s);

    total_reserved_ += slab_bytes;
}

uint64_t CaptureSafeArena::alloc_temp(size_t nbytes, size_t align, cudaStream_t stream) {
    if (align == 0) align = 256;
    std::lock_guard<std::mutex> lock(m_);

    // 캡처 중인 스트림에서 슬랩이 부족하면 -> 금지
    if (is_stream_capturing(stream)) {
        // LIFO에서 적합 블록 있는지 먼저 시도
        for (auto it = lifo_.rbegin(); it != lifo_.rend(); ++it) {
            if (it->size >= nbytes) {
                uint64_t ptr = it->ptr;
                curr_in_use_ += nbytes;
                peak_in_use_ = std::max(peak_in_use_, curr_in_use_);
                lifo_.erase(std::next(it).base()); // reverse_iterator erase
                return ptr;
            }
        }
        // 슬랩 추가가 필요한 경우 -> 캡처 중 금지
        throw std::runtime_error("[CaptureSafeArena] Out of pre-reserved memory during CUDA Graph capture");
    }

    // 비캡처 상황: LIFO 재사용 우선
    for (auto it = lifo_.rbegin(); it != lifo_.rend(); ++it) {
        if (it->size >= nbytes) {
            uint64_t ptr = it->ptr;
            curr_in_use_ += nbytes;
            peak_in_use_ = std::max(peak_in_use_, curr_in_use_);
            lifo_.erase(std::next(it).base());
            return ptr;
        }
    }

    // 슬랩에서 bump
    for (auto& sl : slabs_) {
        size_t off = align_up(sl.bump, align);
        if (off + nbytes <= sl.bytes) {
            uint64_t ptr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(sl.base) + off);
            sl.bump = off + nbytes;
            curr_in_use_ += nbytes;
            peak_in_use_ = std::max(peak_in_use_, curr_in_use_);
            return ptr;
        }
    }

    // 부족하면 새 슬랩 확장 후 재시도
    size_t grow = std::max(nbytes * 2, size_t(16 << 20)); // 16MB 최소
    grow = align_up(grow, 256);
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, grow);
    if (err != cudaSuccess) throw std::runtime_error("CaptureSafeArena::alloc_temp cudaMalloc grow failed");
    slabs_.push_back({p, grow, 0});
    total_reserved_ += grow;

    // 이제 할당
    auto& sl = slabs_.back();
    size_t off = align_up(sl.bump, align);
    uint64_t ptr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(sl.base) + off);
    sl.bump = off + nbytes;
    curr_in_use_ += nbytes;
    peak_in_use_ = std::max(peak_in_use_, curr_in_use_);
    return ptr;
}

void CaptureSafeArena::free_temp(uint64_t ptr, cudaStream_t /*stream*/) {
    std::lock_guard<std::mutex> lock(m_);
    // LIFO push (사이즈 추적이 필요하므로 헤더 없이 단순 push는 곤란)
    // 간편화를 위해: 현재는 "최근 alloc의 크기"를 몰라서 토큰에 사이즈를 싣는 게 정석.
    // MVP: 일단 'unknown size' 방지 위해 상층에서 사이즈를 기억하게 하거나,
    //      여기선 보수적으로 0으로 넣지 말고 API 변경을 권장.
    // ----> 실사용을 위해 alloc_temp 반환을 (ptr << 32 | size_low) 같은 토큰으로 바꾸는 걸 추천.
    // 여기선 단순화: free는 통계만 감소시키고, 재사용 풀에는 넣지 않는다 (메모리 파편화 대신 안전성 택)
    // 필요 시 아래 로직을 토큰 기반으로 교체.
    (void)ptr;
    // no-op reuse. 개발 단계에서는 reset_pool로 프레임마다 리셋 추천.
}

void CaptureSafeArena::reset_pool() {
    std::lock_guard<std::mutex> lock(m_);
    for (auto& sl : slabs_) sl.bump = 0;
    lifo_.clear();
    curr_in_use_ = 0;
}

void CaptureSafeArena::destroy_all() {
    std::lock_guard<std::mutex> lock(m_);
    for (auto& sl : slabs_) {
        if (sl.base) cudaFree(sl.base);
    }
    slabs_.clear();
    lifo_.clear();
    total_reserved_ = peak_in_use_ = curr_in_use_ = 0;
}

AllocStats CaptureSafeArena::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    AllocStats s;
    s.total_reserved = total_reserved_;
    s.peak_in_use    = peak_in_use_;
    s.curr_in_use    = curr_in_use_;
    s.slabs          = int(slabs_.size());
    return s;
}
