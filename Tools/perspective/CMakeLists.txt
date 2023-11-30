add_executable(perspective perspective.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(perspective ${OpenCV_LIBS} compiler_flags)