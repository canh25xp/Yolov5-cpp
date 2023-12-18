#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
namespace Yolo {
struct Object {
    cv::Rect_<float> rect;
    int label {};
    float prob {};
    std::vector<float> mask_feat;
    cv::Mat cv_mask;
};
}