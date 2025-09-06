#!/usr/bin/env bash
set -e
mkdir -p build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc)
ctest --output-on-failure || ./tests/test_basic
