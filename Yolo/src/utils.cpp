#include "yolo/utils.hpp"
#include "object.hpp"
#include "yolo/detector.hpp"
#include "yolo/visualize.hpp"
#include "yolo/general.hpp"
#include "yolo/dataloader.hpp"
#include "yolo/logger.hpp"

#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Yolo {
Utils::Utils() {
    detector = new Detector();
}

Utils::~Utils() {
    delete detector;
}

int Utils::run() {
    Logger::Init();
    if (this->fp32)
        detector->use_fp32();

    if (detector->load(model))
        return -1;

    std::filesystem::path dataPath = this->data;

    get_class_names(class_names, dataPath);

    std::filesystem::path outputPath = project + "/" + name;
    std::filesystem::path inputPath = input;

    if (not inputPath.has_extension()) {
        this->show = false;
        folder(inputPath, outputPath);
        return 0;
    }

    if (isImage(inputPath)) {
        image(inputPath, outputPath);
        return 0;
    }
    if (input == "0" or isVideo(inputPath)) {
        video(inputPath.string());
        return 0;
    }

    LOG_ERROR("input type not supported");
    return 1;
}

void Utils::image(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder) {
    cv::Mat in = cv::imread(inputPath.string());
    std::vector<Object> objects;
    if (dynamic)
        detector->detect_dynamic(in, objects, target_size, prob_threshold, agnostic, max_object);
    else
        detector->detect(in, objects, target_size, prob_threshold, agnostic, max_object);

    std::string fileName = inputPath.filename().string();
    std::string stem = inputPath.stem().string();
    std::string outputPath = outputFolder.string() + "/" + fileName;
    std::string labelsFolder = outputFolder.string() + "/labels";
    std::string labelsPath = labelsFolder + "/" + stem + ".txt";
    std::string cropFolder = outputFolder.string() + "/crop";
    std::string maskFolder = outputFolder.string() + "/mask";
    std::string rotateFolder = outputFolder.string() + "/rotate";
    std::string anglePath = rotateFolder + "/" + "angle.txt";

    const size_t objCount = objects.size();
    LOG_INFO("Objects count = {}\n", objCount);

    int color_index = 0;
    cv::Mat out = in.clone();
    int colorMode = byClass;
    std::string labels; // class-index confident center-x center-y box-width box-height
    std::string contours; // x y contours points
    for (int i = 0; i < objCount; i++) {
        const Object& obj = objects[i];
        if (colorMode == byClass)
            color_index = obj.label;
        if (colorMode == byIndex)
            color_index = i;

        const unsigned char* color = colors[color_index % 80];
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
            draw_label(out, obj.rect, class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%");

        cv::Mat binMask;
        cv::threshold(obj.cv_mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization

        std::vector<cv::Point> contour = mask2segment(obj.cv_mask);
        if (drawContour)
            cv::polylines(out, contour, true, cc, thickness);
        else
            draw_mask(out, obj.cv_mask, color);

        //std::string saveFileName = stem + "_" + std::to_string(i) + "_" + class_names[obj.label] + ".jpg";
        std::string saveFileName = fileName;

        float rotAngle = 0;
        if (rotate) {
            cv::Mat rotated;
            if (contour.size() < 3)
                rotated = in(obj.rect);
            else {
                cv::RotatedRect rr = cv::minAreaRect(contour);
                if (drawMinRect)
                    draw_RotatedRect(out, rr, cv::Scalar(0, 255, 0), thickness);
                rotAngle = getRotatedRectImg(in, rotated, rr);
            }
            cv::utils::fs::createDirectory(rotateFolder);
            std::string rotatePath = rotateFolder + "/" + saveFileName;
            if (show)
                cv::imshow("Rotated", rotated);
            if (save)
                cv::imwrite(rotatePath, rotated);

            std::ofstream angle;
            angle.open(anglePath, std::ios::app);
            angle << stem << " " << std::to_string(rotAngle) << std::endl;
            angle.close();
        }

        if (crop) {
            cv::utils::fs::createDirectory(cropFolder);
            cv::Mat RoI(in, obj.rect); //Region Of Interest
            std::string cropPath = cropFolder + "/" + saveFileName;
            cv::imwrite(cropPath, RoI);
        }

        if (save_mask) {
            cv::utils::fs::createDirectory(maskFolder);
            std::string maskPath = maskFolder + "/" + saveFileName;
            cv::imwrite(maskPath, binMask);
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
        cv::utils::fs::createDirectory(outputFolder.string());
        cv::imwrite(outputPath, out);
        LOG_INFO("\nOutput saved at {}", outputPath);
    }

    if (save_txt) {
        cv::utils::fs::createDirectory(labelsFolder);
        std::ofstream txtFile(labelsPath);
        txtFile << labels << " " << contours;
        txtFile.close();
        LOG_INFO("\nLabels saved at {}", labelsPath);
    }
}

void Utils::folder(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder) {
    LOG_INFO("Auto running on all images in the input folder");
    int count = 0;
    auto tStart = std::chrono::high_resolution_clock::now();
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(inputPath)) {
        std::string path = entry.path().string();
        LOG_INFO("\n------------------------------------------------\n");
        LOG_INFO("{}\n", path);
        if (isImage(path)) {
            count++;
            image(entry.path(), outputFolder);
        }
        else {
            LOG_INFO("skipping non image file");
        }
    }
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tStart).count();
    double average = total / count;
    LOG_INFO("\n------------------------------------------------\n");
    LOG_INFO("{} images processed\n", count);
    LOG_INFO("Total time taken: {} ms\n",total);
    LOG_INFO("Average time taken: {} ms\n", average);
}

void Utils::video(std::string inputPath) {
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

        std::vector<Object> objects;

        cv::Mat frame;
        size_t frameIndex = 0;
        do {
            capture >> frame; //extract frame by frame
            if (dynamic)
                detector->detect_dynamic(frame, objects, target_size, prob_threshold, agnostic, max_object);
            else
                detector->detect(frame, objects, target_size, prob_threshold, agnostic, max_object);
            draw_objects(frame, objects, 0, class_names);
            cv::imshow("Detect", frame);
            if (save) {
                cv::utils::fs::createDirectory("../frame");
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
}