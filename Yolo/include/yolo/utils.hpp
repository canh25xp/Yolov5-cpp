#pragma once
#include <filesystem>
#include <vector>
#include <opencv2/core.hpp>

namespace Yolo {
class Detector;
struct Object;
class Utils {
public:
    Utils();

    ~Utils();

public:
    bool save = false;
    bool drawContour = false;
    bool crop = false;
    bool save_txt = false;
    bool save_mask = false;
    bool rotate = false;
    bool show = false;
    bool dynamic = false;
    bool agnostic = false;
    bool fp32 = false;
    bool noBbox = false;
    bool noLabel = false;
    bool drawMinRect = false;

    int target_size = 640;
    float prob_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_object = 1;
    int padding = 0;
    int thickness = 3;

    std::string project = "runs/idcard";
    std::string name = "exp";
    std::string input = "data/images/sample.jpg";
    std::string model = "weights/yolov5s-seg-idcard-best-2.ncnn";
    std::string data = "data/idcard.yaml";

public:
    int run();

    void video(std::string inputPath);

    void image(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder);

    void folder(const std::filesystem::path& inputFolder, const std::filesystem::path& outputFolder);
private:
    Detector* detector;
    std::vector<std::string> class_names;

};
}