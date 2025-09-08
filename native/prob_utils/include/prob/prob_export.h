#pragma once
#if defined(_WIN32) || defined(_WIN64)
  #if defined(PROB_EXPORTS)
    #define PROB_API __declspec(dllexport)
  #else
    #define PROB_API __declspec(dllimport)
  #endif
#else
  #define PROB_API __attribute__((visibility("default")))
#endif
