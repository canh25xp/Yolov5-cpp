#pragma once

#ifdef NDEBUG
#define TIME_LOG(time)
#else
#define TIME_LOG(time) Timer timer(time)
#endif // NDEBUG

#ifdef BENCHMARK
#define LOG(message)
#else
#define LOG(message) std::cout << message
#endif // _DEBUG

