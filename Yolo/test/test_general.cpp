#include "yolo/general.hpp"
#include <iostream>
#include <string>

using namespace std;
int main() {
    std::string pathStr = "runs/idcard/exp";
    auto newPath = Yolo::increment_path(pathStr, false, "", true);

    cout << newPath;
    return 0;
}