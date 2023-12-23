#pragma once
#include <chrono>

namespace Yolo {
class Timer {
public:
    Timer();
    Timer(double& elapsedTime);
    ~Timer();

    /// @brief Reset timer
    void Reset();

    /// @brief Get Elapsed Time
    /// @return Elapsed Time in milliseconds
    double ElapsedMillis();

    /// @brief Get duration from the last call to itself or from the start
    /// @return Duration in milliseconds
    double Interval();
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start {};
    std::chrono::time_point<std::chrono::high_resolution_clock> m_lastInterval {};
    double& m_elapsedTime;
};
} // namespace Yolo