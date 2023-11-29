add_executable(affineTransform affineTransform.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(affineTransform PRIVATE ${OpenCV_LIBS})