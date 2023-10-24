apt-get update && apt-get upgrade -y
apt-get install -y python3 vim git wget build-essential gcc make

wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.sh
chmod +x cmake-3.21.1-linux-x86_64.sh
mkdir /root/cmake
./cmake-3.21.1-linux-x86_64.sh --skip-license --prefix=/root/cmake

git clone https://github.com/llvm/llvm-project/ /root/llvm-project
cd /root/llvm-project
git checkout tags/llvmorg-12.0.1

mkdir build
cd build
/root/cmake/bin/cmake -DLLVM_BUILD_EXAMPLES=1 -DCLANG_BUILD_EXAMPLES=1 -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_INSTALL_PREFIX="" -DCMAKE_BUILD_TYPE="Release" -G "Unix Makefiles" ../llvm
make
make install
