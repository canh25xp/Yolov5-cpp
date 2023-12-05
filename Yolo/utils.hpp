#pragma once
#include "yolo.hpp"

#include <filesystem>

#ifdef CV_PARSER
#include <opencv2/core/utility.hpp>
#else
#include "parser.hpp"
#endif

enum strategy {
    concatenatedContour = 0,    //concatenate all segments
    largestContour = 1          //select largest segment
};

enum colorMode {
    byClass = 0,                //color object by class index
    byIndex = 1                 //color object by object number index 
};

extern const unsigned char colors[81][3];

extern std::vector<std::string> IMG_FORMATS;
extern std::vector<std::string> VID_FORMATS;

class Utils {
public:
    Utils();

    Utils(int argc, char** argv);

    ~Utils();

public:
    bool save;
    bool drawContour;
    bool crop;
    bool save_txt;
    bool save_mask;
    bool rotate;
    bool show;
    bool dynamic;
    bool agnostic;
    bool fp32;
    bool noBbox;
    bool noLabel;
    bool drawMinRect;

    int target_size;
    float prob_threshold;
    float nms_threshold;
    int max_object;
    
    int padding;
    int thickness;

    std::string project;
    std::string name;
    std::string input;
    std::string model;
    std::string data;

public:
    int run();

    int load(const std::string& model);

    int load(const std::filesystem::path& bin, const std::filesystem::path& param);

    void video(std::string inputPath);

    void image(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder);

    void folder(const std::filesystem::path& inputFolder, const std::filesystem::path& outputFolder);

    void get_class_names(const std::string& data);

    void get_class_names_yaml(const std::string& data_yaml);

    void get_class_names(const std::filesystem::path& data);

    void set_arguments(int argc, char** argv);

private:
    Yolo yolo;
    std::vector<std::string> class_names;

private:

    /// @brief Draw all the objects at once.
    /// This function is a combination of draw_mask, draw_label and cv::rectangle
    /// @param bgr background image to be draw on.
    /// @param objects object vector contain all the detected object in the image
    /// @param colorMode determine the color for each object to be draw ( bounding box and feature mask )
    void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int colorMode = byIndex);

    /// @brief draw color mask on the background image.
    /// @param bgr background image to be draw on.
    /// @param mask gray scale mask
    /// @param color color to be draw on the mask
    void draw_mask(cv::Mat& bgr, const cv::Mat& mask, const unsigned char* color);

    void draw_RotatedRect(cv::Mat& bgr, const cv::RotatedRect& rect, const cv::Scalar& cc, int thickness = 1);

    std::vector<cv::Point> mask2segment(const cv::Mat& mask, int strategy = largestContour);

    void draw_label(cv::Mat& bgr, const cv::Rect2f& rect, std::string label);

    cv::Mat applyMask(const cv::Mat& bgr, const cv::Mat& mask);

    float getRotatedRectImg(const cv::Mat& src, cv::Mat& dst, const cv::RotatedRect& rr);

    bool isImage(const std::string& path);

    bool isImage(const std::filesystem::path& path);

    bool isVideo(const std::string& path);

    bool isVideo(const std::filesystem::path& path);
};