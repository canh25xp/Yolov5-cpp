#pragma once
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/core/mat.hpp>

namespace Yolo {

enum strategy {
    concatenatedContour = 0,    //concatenate all segments
    largestContour = 1          //select largest segment
};

extern std::vector<std::string> IMG_FORMATS;
extern std::vector<std::string> VID_FORMATS;

bool isImage(const std::string& path);

bool isImage(const std::filesystem::path& path);

bool isVideo(const std::string& path);

bool isVideo(const std::filesystem::path& path);

bool isFolder(const std::filesystem::path& path);

bool isURL(const std::string& path);

std::vector<cv::Point> mask2segment(const cv::Mat& mask, int strategy = largestContour);

cv::Mat applyMask(const cv::Mat& bgr, const cv::Mat& mask);

float getRotatedRectImg(const cv::Mat& src, cv::Mat& dst, const cv::RotatedRect& rr);

std::filesystem::path increment_path(const std::filesystem::path& pathStr, bool exist_ok = false, const std::string& sep = "", bool mkdir = false);

std::vector<std::string> getListFileDirs(const std::string& basePath);

} // namespace Yolo
