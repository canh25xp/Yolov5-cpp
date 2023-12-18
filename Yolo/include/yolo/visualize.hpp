#pragma once

#include <ncnn/mat.h>
#include <opencv2/core/mat.hpp>

namespace Yolo {

struct Object;

enum colorMode {
    byClass = 0,                //color object by class index
    byIndex = 1                 //color object by object number index 
};

extern const unsigned char colors[81][3];

/// <summary>
/// Print ncnn Mat value
/// </summary>
/// <param name="mat">NCNN Mat to be print</param>
void mat_print(const ncnn::Mat& mat);

/// <summary>
/// Visualize NCNN Mat. The method first convert ncnn::Mat to cv::Mat.
/// </summary>
/// <param name="title">title of the window</param>
/// <param name="mat">NCNN Mat to be visualize</param>
/// <param name="save">Save output image</param>
void mat_visualize(const char* title, const ncnn::Mat& mat, bool save = 0);

/// <summary>
/// Convert ncnn::Mat to cv::Mat
/// </summary>
/// <param name="mat">input ncnn mat</param>
/// <param name="c">channel index</param>
/// <returns>converted cv mat</returns>
cv::Mat ncnn2cv(const ncnn::Mat& mat, int c = 0);

/// @brief Draw Label to image
/// @param bgr Image to be draw to
/// @param rect Location of the bounding box, the label with be draw on the top left of it.
/// @param label The label text
void draw_label(cv::Mat& bgr, const cv::Rect2f& rect, const std::string& label);

/// @brief Draw the rotated rectangle
/// @param bgr The background image to be draw to
/// @param rr The rotated rectangle
/// @param cc Color of the Rectangle
/// @param thickness Line thickness
/// @param padding Expand the rotated rectangle by the value of padding (pixels)
void draw_RotatedRect(cv::Mat& bgr, const cv::RotatedRect& rr, const cv::Scalar& cc, int thickness = 1, int padding = 0);

/// @brief Draw all the objects at once.
/// This function is a combination of draw_mask, draw_label and cv::rectangle
/// @param bgr background image to be draw on.
/// @param objects object vector contain all the detected object in the image
/// @param colorMode determine the color for each object to be draw ( bounding box and feature mask )
void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int colorMode = byIndex, std::vector<std::string> class_names = {});

/// @brief draw color mask on the background image.
/// @param bgr background image to be draw on.
/// @param mask gray scale mask
/// @param color color to be draw on the mask
void draw_mask(cv::Mat& bgr, const cv::Mat& mask, const unsigned char* color);

} // namespace Yolo
