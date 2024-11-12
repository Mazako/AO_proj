#!/usr/bin/env bash
if [ -d '../cmake-build-release' ]; then
  rm -rf '../cmake-build-release'
fi

cmake -S .. -B '../cmake-build-release'
make -C '../cmake-build-release' 'MillerRabin'