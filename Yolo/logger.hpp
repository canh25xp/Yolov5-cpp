#pragma once

#ifdef NDEBUG
#define TIME_LOG(time)
#define LOG(message)
#else
#define LOG(...) std::cout << __VA_ARGS__
#define TIME_LOG(...) Timer timer(__VA_ARGS__)
#endif // NDEBUG

