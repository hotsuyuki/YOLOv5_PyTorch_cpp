#!/bin/sh

set -v


### LibTorch v1.6.0
wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
unzip ./libtorch-cxx11-abi-shared-with-deps-1.6.0.zip -d ./src/
rm ./libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
mv ./src/libtorch/ ./src/libtorch_v1-6-0/
