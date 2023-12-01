add_executable(boundary boundary.cpp)

target_link_libraries(boundary PRIVATE ${OpenCV_LIBS})