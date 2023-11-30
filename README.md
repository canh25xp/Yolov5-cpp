# Yolov5-cpp

## How to build
### 1. Clone this repo and init submodules
```
git clone https://github.com/canh25xp/yolov5-cpp
cd yolov5-cpp
git submodule update --init
```
### 2. Build with cmake 
The dependencies is manage with [vcpkg](https://github.com/microsoft/vcpkg).
Therefore, you have to have it installed and put the VCPKG_ROOT in the environment variable.
```
cmake --preset vcpkg
cmake --build build --config Release
```
Alternatively :
```
mkdir build
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```
### 3. Pull model files

The model files (*.bin) is tracking with [git-lfs](https://git-lfs.com/).
Therefore if you want to use it, you can run :
```
git lfs pull
```
### 4. Try to run it
```
build\Yolov5-seg\Release\Yolov5-seg.exe --show
```