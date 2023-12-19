#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

namespace Yolo {

enum strategy {
    concatenatedContour = 0,    //concatenate all segments
    largestContour = 1          //select largest segment
};

std::vector<cv::Point> mask2segment(const cv::Mat& mask, int strategy = largestContour);

cv::Mat applyMask(const cv::Mat& bgr, const cv::Mat& mask);

float getRotatedRectImg(const cv::Mat& src, cv::Mat& dst, const cv::RotatedRect& rr);

} // namespace Yolo
