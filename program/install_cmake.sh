#!/usr/bin/env bash
if command -v cmake >/dev/null 2>&1; then
    echo "CMake is installed."
    cmake --version
else
  sudo apt remove cmake
  sudo apt update
  sudo apt install -y software-properties-common lsb-release
  sudo apt clean all
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
  sudo apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
  sudo apt update
  sudo apt install kitware-archive-keyring
  sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
  sudo apt update
  sudo apt install -y cmake
fi


