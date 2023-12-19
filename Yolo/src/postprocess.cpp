#include "yolo/postprocess.hpp"

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Yolo {

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