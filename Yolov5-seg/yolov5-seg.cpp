#include "yolo.hpp"
#include "CLI/CLI.hpp"

int main(int argc, char** argv) {
    Yolo::Utils utils;
    CLI::App parser {"Yolov5 segmentation NCNN"};
    argv = parser.ensure_utf8(argv);

    parser.add_option("--model", utils.model, "model file path")->option_text("model.ncnn");
    parser.add_option("-i,--input,--source", utils.input, "file or folder or 0(webcam)");
    parser.add_option("--data", utils.data, "data file path");
    parser.add_option("--imgsz,--img,--img-size",utils.target_size,"inference size");
    parser.add_option("--conf,--conf-thres",utils.prob_threshold,"confidence threshold");
    parser.add_option("--nms,--iou-thres",utils.nms_threshold,"NMS IoU threshold");
    parser.add_option("--max-obj,--max-det",utils.max_object,"maximum detections per image");
    parser.add_option("--project", utils.project, "save results to project/name");
    parser.add_option("--name", utils.name, "save results to project/name");
    parser.add_option("--line-thickness",utils.thickness,"line thickness for draw (pixels)");
    parser.add_option("--padding",utils.padding,"add padding to min area rect");
    parser.add_flag("--crop", utils.crop, "crop detection");
    parser.add_flag("--save",utils.save,"save results");
    parser.add_flag("--save-txt",utils.save_txt,"save labels");
    parser.add_flag("--save-mask",utils.save_mask,"save mask");
    parser.add_flag("--rotate",utils.rotate,"rotate the detected min area rect");
    parser.add_flag("--show,--view-img",utils.show,"show result");
    parser.add_flag("--dynamic",utils.dynamic,"using dynamic inference");
    parser.add_flag("--agnostic-nms",utils.agnostic,"class-agnostic NMS");
    parser.add_flag("--fp32",utils.fp32,"using 32-bit floating point inference");
    parser.add_flag("--draw-contour",utils.drawContour,"draw contour instead of mask");
    parser.add_flag("--no-bbox",utils.noBbox,"no draw bounding box");
    parser.add_flag("--no-label,--hide-labels",utils.noLabel,"no draw label");
    parser.add_flag("--draw-minrect",utils.drawMinRect,"draw min area rect");

    CLI11_PARSE(parser, argc, argv);
    utils.run();

    return 0;
}