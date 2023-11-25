#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat edge, dst, threshold;
    Mat HSV;
    int dilation_size = 4;

    Mat src = imread("cmt9_back.jpg", 1);
    blur(src, src, Size(2, 2));
    cvtColor(src, HSV, COLOR_BGR2HSV);
    inRange(HSV, Scalar(44, 90, 30), Scalar(76, 255, 255), threshold);
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
    morphologyEx(threshold, dst, MORPH_CLOSE, element);

    int minBound = dst.rows;
    int maxBound = 0;
    int tmp = 0;

    for (int i = 0; i < dst.cols; i++) {
        for (int j = 0; j < dst.rows; j++) {
            if (dst.at<uchar>(j, i) == 0) {
                tmp = j;
                break;
            }
        }
        if (tmp < minBound) minBound = tmp;
        if (tmp > maxBound) maxBound = tmp;
    }

    line(dst, Point(0, minBound), Point(dst.cols, minBound), Scalar(0), 1, 8, 0); //minmum boudary
    line(dst, Point(0, maxBound), Point(dst.cols, maxBound), Scalar(255), 1, 8, 0); // maximum boundary
    line(src, Point(0, minBound), Point(dst.cols, minBound), Scalar(0, 0, 255), 1, 8, 0); //minmum boudary
    line(src, Point(0, maxBound), Point(dst.cols, maxBound), Scalar(0, 0, 255), 1, 8, 0); // maximum boundary

    imshow("source", src);
    imshow("threshold", threshold);
    imshow("dst", dst);
    waitKey(0);
    return 0;
}