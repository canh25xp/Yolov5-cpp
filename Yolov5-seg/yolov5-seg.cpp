#include "yolo.hpp"
#include "CLI/CLI.hpp"

#define GET(...) add_option(__VA_ARGS__)
#define HAS(...) add_flag(__VA_ARGS__)

int main(int argc, char** argv) {
    Yolo::Utils utils;
    CLI::App parser {"Yolov5 segmentation NCNN"};
    argv = parser.ensure_utf8(argv);

    parser.GET("--model",                   utils.model, "model file path");
    parser.GET("-i,--input,--source",       utils.input, "file or folder or 0(webcam)");
    parser.GET("--data",                    utils.data, "data file path");
    parser.GET("--imgsz,--img,--img-size",  utils.target_size, "inference size");
    parser.GET("--conf,--conf-thres",       utils.prob_threshold, "confidence threshold");
    parser.GET("--nms,--iou-thres",         utils.nms_threshold, "NMS IoU threshold");
    parser.GET("--max-obj,--max-det",       utils.max_object, "maximum detections per image");
    parser.GET("--project",                 utils.project, "save results to project/name");
    parser.GET("--name",                    utils.name, "save results to project/name");
    parser.GET("--line-thickness",          utils.thickness, "line thickness for draw (pixels)");
    parser.GET("--padding",                 utils.padding, "add padding to min area rect");
    parser.HAS("--crop",                    utils.crop, "crop detection");
    parser.HAS("--save",                    utils.save, "save results");
    parser.HAS("--save-txt",                utils.save_txt, "save labels");
    parser.HAS("--save-mask",               utils.save_mask, "save mask");
    parser.HAS("--rotate",                  utils.rotate, "rotate the detected min area rect");
    parser.HAS("--show,--view-img",         utils.show, "show result");
    parser.HAS("--dynamic",                 utils.dynamic, "using dynamic inference");
    parser.HAS("--agnostic-nms",            utils.agnostic, "class-agnostic NMS");
    parser.HAS("--fp32",                    utils.fp32, "using 32-bit floating point inference");
    parser.HAS("--draw-contour",            utils.drawContour, "draw contour instead of mask");
    parser.HAS("--no-bbox",                 utils.noBbox, "no draw bounding box");
    parser.HAS("--no-label,--hide-labels",  utils.noLabel, "no draw label");
    parser.HAS("--draw-minrect",            utils.drawMinRect, "draw min area rect");

    CLI11_PARSE(parser, argc, argv);
    utils.run();

    return 0;
}