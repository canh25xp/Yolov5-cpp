add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_20)

set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
target_compile_options(
    compiler_flags
    INTERFACE
    "$<${msvc_cxx}:$<BUILD_INTERFACE:-W1>>"
    "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
)

set(Yolo_Src yolo.cpp utils.cpp layers.cpp parser.cpp timer.cpp pch.cpp)

add_library(Yolo STATIC ${Yolo_Src})

target_include_directories(
    Yolo
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )

# Add(uncomment) the following line if you haven't add opencv_dir to your system enviroment
# set(Opencv_Dir "D:/Lib/opencv/build/install/x64/vc17")
find_package(OpenCV REQUIRED)

# Add(uncomment) the following line if you haven't add ncnn_dir to your system enviroment
# set(ncnn_DIR "D:/Lib/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

# MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
# MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

target_link_libraries(Yolo PUBLIC yaml-cpp ncnn ${OpenCV_LIBS} compiler_flags)
