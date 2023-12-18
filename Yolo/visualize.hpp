#pragma once

#include <ncnn/mat.h>
#include <opencv2/core/mat.hpp>

namespace Yolo {

/// <summary>
/// Print ncnn Mat value
/// </summary>
/// <param name="mat">NCNN Mat to be print</param>
void mat_print(const ncnn::Mat & mat);

/// <summary>
/// Visualize NCNN Mat. The method first convert ncnn::Mat to cv::Mat.
/// </summary>
/// <param name="title">title of the window</param>
/// <param name="mat">NCNN Mat to be visualize</param>
/// <param name="save">Save output image</param>
void mat_visualize(const char* title, const ncnn::Mat & mat, bool save = 0);

/// <summary>
/// Convert ncnn::Mat to cv::Mat
/// </summary>
/// <param name="mat">input ncnn mat</param>
/// <param name="c">channel index</param>
/// <returns>converted cv mat</returns>
cv::Mat ncnn2cv(const ncnn::Mat& mat, int c = 0);

} // namespace Yolo
