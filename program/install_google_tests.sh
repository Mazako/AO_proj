#!/usr/bin/env bash
if [ ! -d  '../third_party' ]; then
  mkdir '../third_party'
fi

if [ -d '../third_party/googletest' ]; then
  echo 'Google Tests is installed'
else
  wget https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  tar -xzf './googletest-1.15.2.tar.gz' -C '../third_party'
  rm './googletest-1.15.2.tar.gz'
  mv '../third_party/googletest-1.15.2' '../third_party/googletest'
  echo 'Installed google tests'
fi