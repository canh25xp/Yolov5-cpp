cmake_minimum_required(VERSION 3.15)
project(Yolov5-cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(yaml-cpp)
add_subdirectory(Yolo)
add_subdirectory(Yolov5-seg)
add_subdirectory(imutils-cpp)
add_subdirectory(Tools)