# Yolov5-cpp

## How to build
### 1. Clone this repo and init submodules
```
git clone https://github.com/canh25xp/yolov5-cpp
cd yolov5-cpp
```
The cmake script will automatically init the submodule so you dont have to `git submodule update --init` anymore
### 2. Config 
The dependencies is manage with [vcpkg](https://github.com/microsoft/vcpkg).
Have it installed and put the VCPKG_ROOT in the environment variable.
Otherwise cmake will attempt to clone vcpkg
```bash
cmake -B build -S . --toolchain=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
# Alternatively :
cmake --preset windows-static
# Or :
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
```
### 3. Build
```
cmake --build build --config Release
```
### 4. Try to run it
The model files (*.bin) is tracking with [git-lfs](https://git-lfs.com/).
Therefore if you want to use it, you can run :
```bash
git lfs pull
# or to pull specific model 
git lfs pull -I weights/yolov5s-seg-idcard-best.ncnn.bin
```
```
build\Yolov5-seg\Release\Yolov5-seg.exe --show --rotate
```
![output](./runs/idcard/exp/sample.jpg)
![rotated](./runs/idcard/exp/rotate/sample.jpg)
