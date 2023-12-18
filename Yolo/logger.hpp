#pragma once

#pragma warning(push, 0)
#include <spdlog/spdlog.h>
#pragma warning(pop)

#ifdef NDEBUG
#define TIME_LOG(...)
#define LOG_TRACE(...)
#define LOG_INFO(...)       Yolo::Logger::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)       Yolo::Logger::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)      Yolo::Logger::GetCoreLogger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...)   Yolo::Logger::GetCoreLogger()->critical(__VA_ARGS__)
#else
#define TIME_LOG(...) Timer timer(__VA_ARGS__)
// Core log macros
#define LOG_TRACE(...)      Yolo::Logger::GetCoreLogger()->trace(__VA_ARGS__)
#define LOG_INFO(...)       Yolo::Logger::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)       Yolo::Logger::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)      Yolo::Logger::GetCoreLogger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...)   Yolo::Logger::GetCoreLogger()->critical(__VA_ARGS__)
#endif // NDEBUG

namespace Yolo {

class Logger {
public:
    static void Init();
    static std::shared_ptr<spdlog::logger>& GetCoreLogger();
private:
    static std::shared_ptr<spdlog::logger> s_CoreLogger;
};

} // namespace Yolo


