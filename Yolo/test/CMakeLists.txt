add_executable(test_logger test_logger.cpp)
target_link_libraries(test_logger PRIVATE Yolo spdlog)

add_executable(test_general test_general.cpp)
target_link_libraries(test_general PRIVATE Yolo compiler_flags)