#include "yolo/general.hpp"
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Yolo {

std::vector<std::string> IMG_FORMATS {"bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm"};
std::vector<std::string> VID_FORMATS {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"};
std::vector<std::string> URL_PREFIXES {"rtsp://", "rtmp://", "http://", "https://"};

bool isImage(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool isImage(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool isVideo(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}

bool isVideo(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}

bool isFolder(const std::filesystem::path& path) {
    return path.has_extension();
}

bool isURL(const std::string& path) {
    for (const auto& prefix : URL_PREFIXES) {
        if (path.compare(0, prefix.length(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

std::filesystem::path increment_path(const std::filesystem::path& pathStr, bool exist_ok, const std::string& sep, bool mkdir) {
    namespace fs = std::filesystem;

    fs::path path(pathStr);

    if (fs::exists(path) && !exist_ok) {
        fs::path base_path, suffix;
        if (fs::is_regular_file(path)) {
            base_path = path.parent_path() / path.stem();
            suffix = path.extension();
        }
        else
            base_path = path;

        for (int n = 2; n < 9999; ++n) {
            fs::path p = base_path;
            if (!sep.empty())
                p += sep;
            
            p += std::to_string(n);
            if (!suffix.empty())
                p += suffix;

            if (!fs::exists(p)) {
                path = p;
                break;
            }
        }
    }

    if (mkdir)
        fs::create_directories(path);

    return path;
}

cv::Mat applyMask(const cv::Mat& bgr, const cv::Mat& mask) {
    cv::Mat binMask;
    cv::threshold(mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization
    cv::Mat maskCopy;
    binMask.convertTo(maskCopy, CV_8U);
    cv::Mat applyMask;
    bgr.copyTo(applyMask, maskCopy);
    return applyMask;
}

std::vector<cv::Point> mask2segment(const cv::Mat& mask, int strategy) {
    cv::Mat binMask;
    cv::threshold(mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization
    cv::Mat maskCopy;
    binMask.convertTo(maskCopy, CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> contour;

    if (!contours.size())
        return contour;

    if (strategy == concatenatedContour) {
        for (std::vector<cv::Point> concatenatedPoints : contours) {
            contour.insert(contour.end(), concatenatedPoints.begin(), concatenatedPoints.end());
        }
    }
    else {
        contour = *std::max_element(contours.begin(), contours.end(),
                                    [] (const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                    return a.size() < b.size();
                                    });
    }

    return contour;
}

float getRotatedRectImg(const cv::Mat& src, cv::Mat& dst, const cv::RotatedRect& rr) {
    float angle = rr.angle;
    float width = rr.size.width;
    float height = rr.size.height;

    if (rr.size.width < rr.size.height) {
        std::swap(width, height);
        angle = angle - 90;
    }

    float radianAngle = -angle * CV_PI / 180;
    // angle += M_PI; // you may want rotate it upsidedown
    float sinA = sin(radianAngle), cosA = cos(radianAngle);
    float data[6] =
    {
        cosA, -sinA, width / 2.0f - cosA * rr.center.x + sinA * rr.center.y,
        sinA, cosA, height / 2.0f - cosA * rr.center.y - sinA * rr.center.x
    };
    cv::Mat affineMatrix(2, 3, CV_32FC1, data);

    /*
    Alternate way to get affineMatrix matrix
    cv::Mat affineMatrix(2, 3, CV_32FC1);
    affineMatrix.at<float>(0, 0) = cosA;
    affineMatrix.at<float>(0, 1) = sinA;
    affineMatrix.at<float>(0, 2) = width / 2.0f - cosA * rr.center.x - sinA * rr.center.y;
    affineMatrix.at<float>(1, 0) = -sinA;
    affineMatrix.at<float>(1, 1) = cosA;
    affineMatrix.at<float>(1, 2) = height / 2.0f - cosA * rr.center.y + sinA * rr.center.x;
    */
    //cv::Mat affineMatrix = cv::getRotationMatrix2D(rr.center, rr.angle, 1.0);
    //cv::warpAffine(src, result, affineMatrix, src.size(), cv::INTER_CUBIC);

    cv::warpAffine(src, dst, affineMatrix, cv::Size2f(width, height), cv::INTER_CUBIC);

    return angle;
}

} // namespace Yolo
