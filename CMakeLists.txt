cmake_minimum_required(VERSION 3.15)
project(Yolov5-cpp VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_TOOLS "well, to build tools" ON)

add_subdirectory(yaml-cpp)
add_subdirectory(Yolo)
add_subdirectory(Yolov5-seg)
add_subdirectory(imutils-cpp)
if(BUILD_TOOLS)
    add_subdirectory(Tools)
endif()