#include "timer.hpp"
#include <iostream>

using namespace std::chrono_literals;
using namespace std::chrono;

namespace Yolo{
Timer::Timer(const char* _task) {
    task = _task;
    start = high_resolution_clock::now();
}

Timer::~Timer() {
    finish = high_resolution_clock::now();
    duration = finish - start;
    std::cout << task << " took " << duration.count() << std::endl;
}
} // namespace Yolo