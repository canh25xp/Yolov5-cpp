#pragma once
#include <vector>
#include <ncnn/mat.h>

namespace Yolo {

struct Size {
    int width;
    int height;
};


struct Anchor {
    ncnn::Mat feature_blob;
    int stride;
    std::vector<Size> size;
};

} // namespace Yolo