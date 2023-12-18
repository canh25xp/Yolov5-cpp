#pragma once
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

    
    /// <summary>
    /// Use 32-bit floating point inference, otherwise use 16-bit floating point
    /// </summary>
    void use_fp32();
    // TODO: is this option really do anything ?

    /// <summary>
    /// Load ncnn model
    /// </summary>
    /// <param name="bin">path to *.bin</param>
    /// <param name="param">path to *param</param>
    /// <returns>0 if successed</returns>
    int load(const char* bin, const char* param);

    /// <summary>
    /// Normal detection
    /// </summary>
    /// <param name="bgr">Input image</param>
    /// <param name="objects">Vector contains detected objects</param>
    /// <param name="target_size">Target size</param>
    /// <param name="prob_threshold">Confident threshold</param>
    /// <param name="nms_threshold">non max suppression threshold</param>
    /// <param name="agnostic">Agnostic</param>
    /// <param name="max">max objects detection</param>
    /// <returns>0 if success</returns>
    int detect(const cv::Mat& bgr, std::vector<Object>& objects, int target_size = 640, float prob_threshold = 0.25f, float nms_threshold = 0.45f, bool agnostic = false, int max = 100);

    /// <summary>
    /// Dynamic detection
    /// </summary>
    /// <param name="bgr">Input image</param>
    /// <param name="objects">Vector contains detected objects</param>
    /// <param name="target_size">Target size</param>
    /// <param name="prob_threshold">Confident threshold</param>
    /// <param name="nms_threshold">non max suppression threshold</param>
    /// <param name="agnostic">Agnostic</param>
    /// <param name="max">max objects detection</param>
    /// <returns>0 if success</returns>
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

    /// <summary>
    /// Apply non max suppression
    /// </summary>
    /// <param name="faceobjects">objects vector</param>
    /// <param name="picked">picked objects index vector</param>
    /// <param name="nms_threshold">non max suppression threshold</param>
    /// <param name="agnostic">agnostic flag</param>
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = true);
};
}