add_executable(perspective perspective.cpp)

target_link_libraries(perspective ${OpenCV_LIBS} compiler_flags)