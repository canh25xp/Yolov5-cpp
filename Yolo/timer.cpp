#include "timer.hpp"
#include <chrono>

using namespace std::chrono;

namespace Yolo{
Timer::Timer(double& elapsedTime) : m_elapsedTime(elapsedTime) {
    Reset();
}

void Timer::Reset() {
    m_start = high_resolution_clock::now();
    m_lastInterval = m_start;
    m_elapsedTime = 0.0;
}

double Timer::ElapsedMillis() {
    return duration<double, std::milli> (high_resolution_clock::now() - m_start).count(); //duration in milliseconds
}

double Timer::Interval() {
    time_point<high_resolution_clock> now = high_resolution_clock::now();
    double interval = duration<double, std::milli>(now - m_lastInterval).count();
    m_lastInterval = now;
    return interval;
}

Timer::~Timer() {
    m_elapsedTime = ElapsedMillis();
}
} // namespace Yolo