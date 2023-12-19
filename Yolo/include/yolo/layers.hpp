#pragma once
#include <ncnn/layer.h>

namespace Yolo{
void Matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob);

void Slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis);

void Interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out);

void Reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d);

void Sigmoid(ncnn::Mat& bottom);
} // namespace Yolo