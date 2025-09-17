#!/usr/bin/env bash
set -euo pipefail
build_type=${1:-Release}
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=${build_type} ..
cmake --build . -j
