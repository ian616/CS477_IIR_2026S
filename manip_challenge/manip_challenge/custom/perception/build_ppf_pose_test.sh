#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
g++ -std=c++17 -O2 ppf_pose_test.cpp -o ppf_pose_test \
  $(pkg-config --cflags --libs opencv4) \
  -lassimp

echo "built $(pwd)/ppf_pose_test"
