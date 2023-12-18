#pragma once

#include <filesystem>

#include <ncnn/net.h>
#include <opencv2/core/mat.hpp>

#include "timer.hpp"

#ifdef NDEBUG
#define TIME_LOG(name)
#else
#define TIME_LOG(name) Timer timer(name)
#endif // NDEBUG

#ifdef BENCHMARK
#define LOG(message)
#else
#define LOG(message) std::cout << message
#endif // _DEBUG

// TODO: change doxygen comment style
namespace Yolo {

struct Object;
struct Anchor;

class Detector {
public:
    Detector();

    ~Detector();

        
    /// @brief Use 32-bit floating point inference, otherwise use 16-bit floating point
    void use_fp32();
    // TODO: is this option really do anything ?

    /// @brief This load method assumed *.bin and *.param file have the same name and in the same folder. For example : yolov5-seg.bin, yolov5-seg.param
    /// @param model name of the model without extension
    /// @return 0 if successfully load, -1 if fail
    int load(const std::string& model);

    int load(const std::filesystem::path& bin, const std::filesystem::path& param);

    /// @brief Load ncnn model bin and param
    /// @param bin path to *.bin
    /// @param param path to *.param
    /// @return 0 if successed
    int load(const char* bin, const char* param);

    /// @brief Normal detection
    /// @param bgr Input image
    /// @param objects Vector contains detected objects
    /// @param target_size Target size
    /// @param prob_threshold Confident threshold
    /// @param nms_threshold Non max suppression threshold
    /// @param agnostic Agnostic
    /// @param max Max objects detection
    /// @return 0
    int detect(const cv::Mat& bgr, std::vector<Object>& objects, int target_size = 640, float prob_threshold = 0.25f, float nms_threshold = 0.45f, bool agnostic = false, int max = 100);

    /// @brief Dynamic detection
    /// @param bgr Input image
    /// @param objects Vector contains detected objects
    /// @param target_size Target size
    /// @param prob_threshold Confident threshold
    /// @param nms_threshold Non max suppression threshold
    /// @param agnostic Agnostic
    /// @param max Max objects detection
    /// @return 0
    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects, int target_size = 640, float prob_threshold = 0.25f, float nms_threshold = 0.45f, bool agnostic = false, int max = 100);
    // TODO: some how refactoring these two functions

    void get_blob_name(const char* in, const char* out, const char* out1, const char* out2, const char* out3, const char* seg);

private:
    ncnn::Net net;

    const char* in_blob   = "in0";
    const char* out_blob  = "out0";
    const char* out1_blob = "out1";
    const char* out2_blob = "out2";
    const char* out3_blob = "out3";
    const char* seg_blob  = "seg";

private:
    void decode_mask(const ncnn::Mat& mask_feat,
                     const int& img_w, const int& img_h,
                     const ncnn::Mat& mask_proto,
                     const ncnn::Mat& in_pad,
                     const int& wpad, const int& hpad,
                     ncnn::Mat& mask_pred_result);

    inline float intersection_area(const Object& a, const Object& b);

    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

    void qsort_descent_inplace(std::vector<Object>& faceobjects);

    float sigmoid(float x);

    float relu(float x);
    
    void generate_proposals(const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);
    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);
    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);
    void generate_proposals(Anchor anchor, float prob_threshold, std::vector<Object>& objects);

    /// @brief Apply non max suppression
    /// @param faceobjects objects vector
    /// @param picked picked objects index vector
    /// @param nms_threshold Non max suppression threshold
    /// @param agnostic
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = true);
};
}