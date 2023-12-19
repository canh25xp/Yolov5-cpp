#include "yolo/general.hpp"
#include <imutils/convenience.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
int main() {
    // Test increment_path
    std::string pathStr = "runs/idcard/exp";
    auto newPath = Yolo::increment_path(pathStr, false, "", true);

    cout << newPath;

    // Test URL
    std::string url = "https://th.bing.com/th/id/OIP.r5cOR1Vrm9lorAvaACKOsAHaEK?rs=1&pid=ImgDetMain";

    cout << Yolo::isURL(url);

    auto img = imutils::urlToImager(url);
    cv::imshow("image from url", img);
    cv::waitKey();

    return 0;
}