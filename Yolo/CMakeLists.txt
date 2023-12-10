set(Yolo_Src utils.cpp layers.cpp parser.cpp timer.cpp detector.cpp)

set(CMAKE_DEBUG_POSTFIX d)

add_library(Yolo ${Yolo_Src})

target_include_directories(
    Yolo
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )

# Add(uncomment) the following line if you haven't add ncnn_dir to your system enviroment
# set(ncnn_DIR "D:/Lib/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn CONFIG REQUIRED)

# MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
# MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")
set_target_properties(Yolo PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

target_link_libraries(Yolo 
    PUBLIC ${OpenCV_LIBS} compiler_flags
    PRIVATE ncnn spdlog yaml-cpp)
