#pragma once

#if defined(_WIN32) || defined(_WIN64)
  #if defined(NORMAL_EXPORTS)
    #define NORMAL_API __declspec(dllexport)
  #else
    #define NORMAL_API __declspec(dllimport)
  #endif
#else
  #define NORMAL_API __attribute__((visibility("default")))
#endif
