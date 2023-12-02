set(Yolo_Src yolo.cpp utils.cpp layers.cpp parser.cpp timer.cpp pch.cpp)

add_library(Yolo ${Yolo_Src})

target_include_directories(
    Yolo
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )

# MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
# MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

target_link_libraries(Yolo PUBLIC yaml-cpp ncnn ${OpenCV_LIBS} compiler_flags)
