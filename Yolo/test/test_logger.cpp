#include "yolo/logger.hpp"
#include <iostream>
#include <string>

using namespace std;
int main() {
    Yolo::Logger::Init();
    std::string info = "rachel";
    LOG_TRACE("hi {}", info);
    LOG_INFO("lmao");
    LOG_WARN("bruh");
    LOG_ERROR("ohno");
    LOG_CRITICAL("fuck");
    
    return 0;
}