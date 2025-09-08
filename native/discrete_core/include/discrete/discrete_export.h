#pragma once
#if defined(_WIN32) || defined(_WIN64)
  #if defined(DISCRETE_EXPORTS)
    #define DISCRETE_API __declspec(dllexport)
  #else
    #define DISCRETE_API __declspec(dllimport)
  #endif
#else
  #define DISCRETE_API __attribute__((visibility("default")))
#endif
