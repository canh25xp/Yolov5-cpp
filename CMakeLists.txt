cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW) # enable MSVC_RUNTIME_LIBRARY target property
project(Yolov5-cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_TOOLS "well, to build tools" ON)
option(WARNING "Turn on all warning" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Add(uncomment) the following line if you haven't add opencv_dir to your system enviroment
# set(Opencv_Dir "D:/Lib/opencv/build/install/x64/vc17")
find_package(OpenCV REQUIRED)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_20)

if(WARNING)
    set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
    set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
    target_compile_options(
        compiler_flags
        INTERFACE
        "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
        "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
    )
endif()

add_subdirectory(ThirdParty/yaml-cpp)
add_subdirectory(ThirdParty/spdlog)
add_subdirectory(Yolo)
add_subdirectory(ThirdParty/CLI11)
add_subdirectory(Yolov5-seg)
add_subdirectory(ThirdParty/imutils-cpp)
if(BUILD_TOOLS)
    add_subdirectory(Tools)
endif()