#include "yolo.hpp"
#include "parser.hpp"

int main(int argc, char** argv) {
    Yolo::Utils utils;

    Parser parser(argc, argv);
    utils.model                    = parser.get("--model", "weights/yolov5s-seg-idcard-best-2.ncnn");
    utils.data                     = parser.get("--data", "data/idcard.yaml");
    utils.input                    = parser.get("--source", "data/images/sample.jpg");
    utils.project                  = parser.get("--project", "runs/idcard");
    utils.name                     = parser.get("--name", "exp");
    utils.crop                     = parser.has("--crop");
    utils.save                     = parser.has("--save");
    utils.save_txt                 = parser.has("--save-txt");
    utils.save_mask		           = parser.has("--save-mask");
    utils.rotate		           = parser.has("--rotate");
    utils.show                     = parser.has("--show");
    utils.dynamic                  = parser.has("--dynamic");
    utils.agnostic                 = parser.has("--agnostic");
    utils.target_size              = parser.get("--size", 640);
    utils.prob_threshold           = parser.get("--conf", 0.25f);
    utils.nms_threshold            = parser.get("--nms", 0.45f);
    utils.max_object               = parser.get("--max-obj", 1);
    utils.fp32                     = parser.has("--fp32");
    utils.drawContour              = parser.has("--draw-contour");
    utils.noBbox                   = parser.has("--no-bbox");
    utils.noLabel                  = parser.has("--no-label");
    utils.drawMinRect              = parser.has("--draw-minrect");
    utils.thickness                = parser.get("--line-thickness", 3);
    utils.padding                  = parser.get("--padding", 0);
    utils.run();

    return 0;
}