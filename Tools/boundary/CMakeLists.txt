add_executable(boundary boundary.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(boundary PRIVATE ${OpenCV_LIBS})