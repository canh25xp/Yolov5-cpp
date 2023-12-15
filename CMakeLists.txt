cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW) # enable MSVC_RUNTIME_LIBRARY target property

include(cmake/AddGitSubmodule.cmake) # To use add_submodule() and deinit_submodule() function
include(FetchContent)

# IF not find vcpkg install locally, add vcpkg as submodule and set vcpkg toolchain file
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    FetchContent_Declare(
        vcpkg
        GIT_REPOSITORY https://github.com/microsoft/vcpkg
        SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/vcpkg"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    FetchContent_MakeAvailable(vcpkg)

    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

project(Yolov5-cpp VERSION 0.1.0 LANGUAGES C CXX)


option(BUILD_TOOLS "well, to build tools" ON)
option(WARNING "Turn on all warning" OFF)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Add(uncomment) the following line if you haven't add opencv_dir to your system enviroment
# set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
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

add_submodule(ThirdParty/yaml-cpp)
add_submodule(ThirdParty/spdlog)
add_submodule(ThirdParty/CLI11)
add_submodule(ThirdParty/imutils-cpp)

add_subdirectory(Yolo)
add_subdirectory(Yolov5-seg)
if(BUILD_TOOLS)
    add_subdirectory(Tools)
endif()