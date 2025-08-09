// logging_config.h
#pragma once
#include <cstdio>

#ifndef VERBOSE
// 0: 거의 침묵, 1: 에폭/요약, 2: 레이어 shape 흐름, 3: 값 샘플 덤프
#define VERBOSE 1
#endif

// 커널 내부 printf on/off (느리니 평소엔 0)
#ifndef DEBUG_KERNEL
#define DEBUG_KERNEL 0
#endif

// 호스트측 로깅
#define LOGV(LVL, ...) do { if (VERBOSE >= (LVL)) { std::printf(__VA_ARGS__); } } while(0)

// 커널측 로깅
#if DEBUG_KERNEL
  #define KPRINTF(...) printf(__VA_ARGS__)
#else
  #define KPRINTF(...) do{}while(0)
#endif
