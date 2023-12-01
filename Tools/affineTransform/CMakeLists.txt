add_executable(affineTransform affineTransform.cpp)

target_link_libraries(affineTransform PRIVATE ${OpenCV_LIBS})