#include "visualize.hpp"
#include "object.hpp"

#include <ncnn/mat.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Yolo {

const unsigned char colors[81][3] = {
    {56,  0,   255},
    {226, 255, 0},
    {0,   94,  255},
    {0,   37,  255},
    {0,   255, 94},
    {255, 226, 0},
    {0,   18,  255},
    {255, 151, 0},
    {170, 0,   255},
    {0,   255, 56},
    {255, 0,   75},
    {0,   75,  255},
    {0,   255, 169},
    {255, 0,   207},
    {75,  255, 0},
    {207, 0,   255},
    {37,  0,   255},
    {0,   207, 255},
    {94,  0,   255},
    {0,   255, 113},
    {255, 18,  0},
    {255, 0,   56},
    {18,  0,   255},
    {0,   255, 226},
    {170, 255, 0},
    {255, 0,   245},
    {151, 255, 0},
    {132, 255, 0},
    {75,  0,   255},
    {151, 0,   255},
    {0,   151, 255},
    {132, 0,   255},
    {0,   255, 245},
    {255, 132, 0},
    {226, 0,   255},
    {255, 37,  0},
    {207, 255, 0},
    {0,   255, 207},
    {94,  255, 0},
    {0,   226, 255},
    {56,  255, 0},
    {255, 94,  0},
    {255, 113, 0},
    {0,   132, 255},
    {255, 0,   132},
    {255, 170, 0},
    {255, 0,   188},
    {113, 255, 0},
    {245, 0,   255},
    {113, 0,   255},
    {255, 188, 0},
    {0,   113, 255},
    {255, 0,   0},
    {0,   56,  255},
    {255, 0,   113},
    {0,   255, 188},
    {255, 0,   94},
    {255, 0,   18},
    {18,  255, 0},
    {0,   255, 132},
    {0,   188, 255},
    {0,   245, 255},
    {0,   169, 255},
    {37,  255, 0},
    {255, 0,   151},
    {188, 0,   255},
    {0,   255, 37},
    {0,   255, 0},
    {255, 0,   170},
    {255, 0,   37},
    {255, 75,  0},
    {0,   0,   255},
    {255, 207, 0},
    {255, 0,   226},
    {255, 245, 0},
    {188, 255, 0},
    {0,   255, 18},
    {0,   255, 75},
    {0,   255, 151},
    {255, 56,  0},
    {245, 255, 0}
};

void mat_print(const ncnn::Mat& mat) {
    for (int q = 0; q < mat.c; q++) {
        const float* ptr = mat.channel(q);
        for (int z = 0; z < mat.d; z++) {
            for (int y = 0; y < mat.h; y++) {
                for (int x = 0; x < mat.w; x++) {
                    printf("%f ", ptr[x]);
                }
                ptr += mat.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

void mat_visualize(const char* title, const ncnn::Mat& mat, bool save) {
    std::vector<cv::Mat> normed_feats(mat.c);

    for (int i = 0; i < mat.c; i++) {
        cv::Mat tmp(mat.h, mat.w, CV_32FC1, (void*) (const float*) mat.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y = 0; y < mat.h; y++) {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x = 0; x < mat.w; x++) {
                float v = tp[x];
                if (v != v) {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }
                sp += 3;
            }
        }
        if (!save) {
            cv::imshow(title, normed_feats[i]);
            cv::waitKey();
        }
    }

    if (save) {
        int tw = mat.w < 10 ? 32 : mat.w < 20 ? 16 : mat.w < 40 ? 8 : mat.w < 80 ? 4 : mat.w < 160 ? 2 : 1;
        int th = (mat.c - 1) / tw + 1;

        cv::Mat show_map(mat.h * th, mat.w * tw, CV_8UC3);
        show_map = cv::Scalar(127);

        // tile
        for (int i = 0; i < mat.c; i++) {
            int ty = i / tw;
            int tx = i % tw;

            normed_feats[i].copyTo(show_map(cv::Rect(tx * mat.w, ty * mat.h, mat.w, mat.h)));
        }
        cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
        //cv::imshow(title, show_map);
        //cv::waitKey();
        cv::imwrite("masks.jpg", show_map);
    }
}

cv::Mat ncnn2cv(const ncnn::Mat& mat, int c) {
    std::vector<cv::Mat> normed_feats(mat.c);
    for (int i = 0; i < mat.c; i++) {
        cv::Mat tmp(mat.h, mat.w, CV_32FC1, (void*) (const float*) mat.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y = 0; y < mat.h; y++) {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x = 0; x < mat.w; x++) {
                float v = tp[x];
                if (v != v) {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }
                sp += 3;
            }
        }
    }
    return normed_feats[c];
}

void draw_label(cv::Mat& bgr, const cv::Rect2f& rect,const std::string& label) {
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = rect.x;
    int y = rect.y - label_size.height - baseLine;
    if (y < 0)
        y = 0;
    if (x + label_size.width > bgr.cols)
        x = bgr.cols - label_size.width;

    cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
    cv::putText(bgr, label, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
}

void draw_RotatedRect(cv::Mat& bgr, const cv::RotatedRect& rr, const cv::Scalar& cc, int thickness, int padding) {
    cv::Point2f vertices[4];

    if (padding != 0) {
        auto new_h = rr.size.height + padding;
        auto new_w = rr.size.width + padding;
        cv::RotatedRect padding_rect(rr.center, cv::Size2f(new_w, new_h), rr.angle);
        padding_rect.points(vertices);
    }

    else
        rr.points(vertices);

    for (int i = 0; i < 4; i++)
        cv::line(bgr, vertices[i], vertices[(i + 1) % 4], cc, thickness);
}

void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int colorMode, std::vector<std::string> class_names) {
    size_t objCount = objects.size();
    // LOG("Objects count = " << objCount << std::endl);

    int color_index = 0;
    for (size_t i = 0; i < objCount; i++) {
        const Object& obj = objects[i];

        char line[256];
        //class-index confident center-x center-y box-width box-height
        sprintf(line, "%i %f %i %i %i %i", obj.label, obj.prob, (int) round(obj.rect.tl().x), (int) round(obj.rect.tl().y), (int) round(obj.rect.br().x), (int) round(obj.rect.br().y));

        // LOG(line << std::endl);

        if (colorMode == byClass)
            color_index = obj.label;

        const unsigned char* color = colors[color_index];
        cv::Scalar cc(color[0], color[1], color[2]);

        if (colorMode == byIndex)
            color_index = i;

        cv::rectangle(bgr, obj.rect, cc, 1);

        std::string text = class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%";

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        draw_mask(bgr, obj.cv_mask, color);
    }
}

void draw_mask(cv::Mat& bgr, const cv::Mat& mask, const unsigned char* color) {
    cv::Mat binMask;
    cv::threshold(mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization
    for (int y = 0; y < bgr.rows; y++) {
        uchar* image_ptr = bgr.ptr(y);
        const float* mask_ptr = binMask.ptr<float>(y);
        for (int x = 0; x < bgr.cols; x++) {
            if (mask_ptr[x]) {
                image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[0] * 0.5);
                image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[2] * 0.5);
            }
            image_ptr += 3;
        }
    }
}

} // namespace Yolo
