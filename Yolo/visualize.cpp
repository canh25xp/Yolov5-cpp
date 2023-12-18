#include "visualize.hpp"

#include <ncnn/mat.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Yolo {
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

} // namespace Yolo
