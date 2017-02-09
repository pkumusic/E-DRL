#!/bin/bash -x

sudo apt-get install cmake
sudo apt-get install g++
sudo apt-get install zlib1g-dev
sudo apt-get install git

git clone https://github.com/miyosuda/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF .

make -j 2

pip install .

