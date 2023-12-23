#include "yolo/yolo.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <CLI/CLI.hpp>

#define GET(...) add_option(__VA_ARGS__)
#define HAS(...) add_flag(__VA_ARGS__)

bool save = false;
bool drawContour = false;
bool crop = false;
bool save_txt = false;
bool save_mask = false;
bool rotate = false;
bool show = false;
bool dynamic = false;
bool agnostic = false;
bool half = true;
bool noBbox = false;
bool noLabel = false;
bool drawMinRect = false;
bool exist_ok = false;

int target_size = 640;
float prob_threshold = 0.25f;
float nms_threshold = 0.45f;
int max_object = 1;
int padding = 0;
int thickness = 3;

std::string project = "runs/idcard";
std::string name    = "exp";
std::string source  = "data/images/sample.jpg";
std::string model   = "weights/yolov5s-seg-idcard-best-2.ncnn";
std::string data    = "data/idcard.yaml";

std::vector<std::string> class_names;

Yolo::Detector detector;

int run();
void video(std::string inputPath);

int main(int argc, char** argv) {
    Yolo::Logger::Init();
    CLI::App parser {"Yolov5 segmentation NCNN"};
    argv = parser.ensure_utf8(argv);

    parser.GET("--model",                   model, "model file path");
    parser.GET("-i,--input,--source",       source, "file or folder or 0(webcam)");
    parser.GET("--data",                    data, "data file path");
    parser.GET("--imgsz,--img,--img-size",  target_size, "inference size");
    parser.GET("--conf,--conf-thres",       prob_threshold, "confidence threshold");
    parser.GET("--nms,--iou-thres",         nms_threshold, "NMS IoU threshold");
    parser.GET("--max-obj,--max-det",       max_object, "maximum detections per image");
    parser.GET("--project",                 project, "save results to project/name");
    parser.GET("--name",                    name, "save results to project/name");
    parser.GET("--line-thickness",          thickness, "line thickness for draw (pixels)");
    parser.GET("--padding",                 padding, "add padding to min area rect");
    parser.HAS("--crop",                    crop, "crop detection");
    parser.HAS("--save",                    save, "save results");
    parser.HAS("--save-txt",                save_txt, "save labels");
    parser.HAS("--save-mask",               save_mask, "save mask");
    parser.HAS("--rotate",                  rotate, "rotate the detected min area rect");
    parser.HAS("--show,--view-img",         show, "show result");
    parser.HAS("--dynamic",                 dynamic, "using dynamic inference");
    parser.HAS("--agnostic-nms",            agnostic, "class-agnostic NMS");
    parser.HAS("--half,!--no-half",         half, "use FP16 half-precision inference");
    parser.HAS("--draw-contour",            drawContour, "draw contour instead of mask");
    parser.HAS("--no-bbox",                 noBbox, "no draw bounding box");
    parser.HAS("--no-label,--hide-labels",  noLabel, "no draw label");
    parser.HAS("--draw-minrect",            drawMinRect, "draw min area rect");
    parser.HAS("--exist-ok",                exist_ok, "overide the already existing project/name, do not increment");

    CLI11_PARSE(parser, argc, argv);
    run();

    return 0;
}

int run() {
    bool is_url = Yolo::isURL(source);
    bool is_folder = Yolo::isFolder(source);
    bool is_image = Yolo::isImage(source);
    bool is_video = Yolo::isVideo(source);
    bool is_webcam = true ? (source == "0") : false;

    // Directories
    std::filesystem::path save_dir = Yolo::increment_path(std::filesystem::path(project) / name, exist_ok, "", save);
    if (save_txt)
        std::filesystem::create_directory(save_dir / "labels");

    if (rotate)
        std::filesystem::create_directory(save_dir / "rotate");

    if (save_mask)
        std::filesystem::create_directory(save_dir / "mask");
        

    // Load model weight and param
    if (detector.load(model, half))
        return -1;

    std::vector<std::filesystem::path> paths {};

    // Load data
    if (is_webcam) {
        // TODO: implement this
        return -1;
    }
    else if (is_video) {
        // TODO: implement this
        return -1;
    }
    else if (is_url){
        // TODO: implement this
        return -1;
    }
    else if (is_image){
        paths.push_back(source);
    }
    else if (is_folder) {
        paths = Yolo::getListFileDirs(source); // Get Images dirs in folder
    }

    std::filesystem::path dataPath = data;
    Yolo::get_class_names(class_names, dataPath);

    // Run inference
    int count = paths.size();
    LOG_INFO("Found {} image(s) in folder", count);
    auto tStart = std::chrono::high_resolution_clock::now();
    for (const auto& path : paths) {
        std::vector<Yolo::Object> objects;
        cv::Mat in = cv::imread(path.string());
        if (dynamic)
            detector.detect_dynamic(in, objects, target_size, prob_threshold, agnostic, max_object);
        else
            detector.detect(in, objects, target_size, prob_threshold, agnostic, max_object);

        std::string fileName = path.filename().string();
        std::string stem = path.stem().string();
        std::filesystem::path save_path = save_dir / path.filename();
        std::filesystem::path txt_path = save_dir / "labels" / path.stem() += ".txt" ;
        std::filesystem::path rotate_path = save_dir / "rotate" / path.filename();
        std::filesystem::path mask_path = save_dir / "mask" / path.filename();
        std::filesystem::path angle_path = save_dir / "rotate" / "angle.txt";
        std::filesystem::path crop_path = save_dir / "crop" / path.filename();

        const size_t objCount = objects.size();
        LOG_INFO("Objects count = {}\n", objCount);

        int color_index = 0;
        cv::Mat out = in.clone();
        int colorMode = Yolo::colorMode::byClass;
        std::string labels; // class-index confident center-x center-y box-width box-height
        std::string contours; // x y contours points
        for (int i = 0; i < objCount; i++) {
            const auto& obj = objects[i];
            if (colorMode == Yolo::colorMode::byClass)
                color_index = obj.label;
            if (colorMode == Yolo::colorMode::byIndex)
                color_index = i;

            const unsigned char* color = Yolo::colors[color_index % 80];
            cv::Scalar cc(color[0], color[1], color[2]);

            char line[256];
            //class-index confident center-x center-y box-width box-height
            sprintf(line, "%i %f %i %i %i %i", obj.label, obj.prob, (int) round(obj.rect.tl().x), (int) round(obj.rect.tl().y), (int) round(obj.rect.br().x), (int) round(obj.rect.br().y));
            labels.append(line);
            if (i != objCount - 1)
                labels.append("\n");

            if (!noBbox)
                cv::rectangle(out, obj.rect, cc, thickness);

            if (!noLabel)
                Yolo::draw_label(out, obj.rect, class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%");

            cv::Mat binMask;
            cv::threshold(obj.cv_mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization

            std::vector<cv::Point> contour = Yolo::mask2segment(obj.cv_mask);
            if (drawContour)
                cv::polylines(out, contour, true, cc, thickness);
            else
                Yolo::draw_mask(out, obj.cv_mask, color);

            //std::string saveFileName = stem + "_" + std::to_string(i) + "_" + class_names[obj.label] + ".jpg";
            // std::string saveFileName = fileName;

        float rotAngle = 0;
        if (rotate) {
            cv::Mat rotated;
            if (contour.size() < 3)
                rotated = in(obj.rect);
            else {
                cv::RotatedRect rr = cv::minAreaRect(contour);
                if (drawMinRect)
                    Yolo::draw_RotatedRect(out, rr, cv::Scalar(0, 255, 0), thickness);
                rotAngle = Yolo::getRotatedRectImg(in, rotated, rr);
            }

            if (show)
                cv::imshow("Rotated", rotated);
            if (save)
                cv::imwrite(rotate_path.string(), rotated);

                std::ofstream angle;
                angle.open(angle_path, std::ios::app);
                angle << stem << " " << std::to_string(rotAngle) << std::endl;
                angle.close();
            }

            if (crop) {
                cv::Mat RoI(in, obj.rect); //Region Of Interest
                cv::imwrite(crop_path.string(), RoI);
            }

            if (save_mask) {
                cv::imwrite(mask_path.string(), binMask);
            }

            if (save_txt) {
                for (auto& point : contour) {
                    contours.append(cv::format("%i %i ", point.x, point.y));
                }
            }
        }

        LOG_INFO(labels);

        if (show) {
            cv::imshow("Detect", out);
            cv::waitKey();
        }

        if (save) {
            std::filesystem::create_directory(save_dir.string());
            cv::imwrite(save_path.string(), out);
            LOG_INFO("\nOutput saved at {}", save_path.string());
        }

        if (save_txt) {
            std::ofstream txtFile(txt_path);
            txtFile << labels << " " << contours;
            txtFile.close();
            LOG_INFO("\nLabels saved at {}", txt_path.string());
        }
    }
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tStart).count();
    double average = total / count;
    LOG_INFO("\n------------------------------------------------\n");
    LOG_INFO("{} images processed\n", count);
    LOG_INFO("Total time taken: {} ms\n", total);
    LOG_INFO("Average time taken: {} ms\n", average);

    return 0;
}

void video(std::string inputPath) {
    cv::VideoCapture capture;
    if (inputPath == "0") {
        capture.open(0);
    }
    else {
        capture.open(inputPath);
    }
    if (capture.isOpened()) {
        LOG_INFO("Object Detection Started....\n");
        LOG_INFO("Press q or esc to stop\n");

        std::vector<Yolo::Object> objects;

        cv::Mat frame;
        size_t frameIndex = 0;
        do {
            capture >> frame; //extract frame by frame
            if (dynamic)
                detector.detect_dynamic(frame, objects, target_size, prob_threshold, agnostic, max_object);
            else
                detector.detect(frame, objects, target_size, prob_threshold, agnostic, max_object);
            Yolo::draw_objects(frame, objects, 0, class_names);
            cv::imshow("Detect", frame);
            if (save) {
                std::filesystem::create_directory("../frame");
                std::string saveFileName = "../frame/" + std::to_string(frameIndex) + ".jpg";
                cv::imwrite(saveFileName, frame);
                frameIndex++;
            }

            char key = (char) cv::pollKey();

            if (key == 27 || key == 'q' || key == 'Q') // Press q or esc to exit from window
                break;
        } while (!frame.empty());
    }
    else {
        LOG_ERROR("Could not Open Camera/Video");
    }
}