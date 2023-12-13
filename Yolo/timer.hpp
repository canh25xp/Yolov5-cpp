#pragma once
#include <chrono>
#include <string>

namespace Yolo {
struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start{}, finish{};
    std::chrono::duration<double, std::milli> duration{}; //duration in milliseconds
    std::string task;
    Timer(const char* _task);
    ~Timer();
};
} // namespace Yolo