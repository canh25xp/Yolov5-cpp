add_library(Yolo STATIC yolo.cpp utils.cpp layers.cpp parser.cpp timer.cpp pch.cpp)

# target_include_directories(
#     Yolo
#     INTERFACE
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#     )

target_include_directories(Yolo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
# set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
# target_compile_options(
#     Yolo
#     INTERFACE
#     "$<${msvc_cxx}:$<BUILD_INTERFACE:-W1>>"
#     "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
# )

find_package(OpenCV REQUIRED)

# Add(uncomment) the following line if you haven't add ncnn_dir to your system enviroment
# set(ncnn_DIR "D:/Lib/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
# set(Opencv_Dir "D:/Lib/opencv/build/install/x64/vc17")

find_package(ncnn REQUIRED)

# MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
# MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

target_link_libraries(Yolo PUBLIC yaml-cpp ncnn ${OpenCV_LIBS})
