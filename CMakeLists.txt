cmake_minimum_required(VERSION 3.15)
project(Yolov5-cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_TOOLS "well, to build tools" ON)

# Add(uncomment) the following line if you haven't add opencv_dir to your system enviroment
# set(Opencv_Dir "D:/Lib/opencv/build/install/x64/vc17")
find_package(OpenCV REQUIRED)

# Add(uncomment) the following line if you haven't add ncnn_dir to your system enviroment
# set(ncnn_DIR "D:/Lib/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

add_subdirectory(yaml-cpp)
add_subdirectory(Yolo)
add_subdirectory(Yolov5-seg)
add_subdirectory(imutils-cpp)
if(BUILD_TOOLS)
    add_subdirectory(Tools)
endif()