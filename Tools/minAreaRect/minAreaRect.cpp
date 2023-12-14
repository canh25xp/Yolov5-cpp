#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main() {
    float angle = 0;
    Mat image(400, 400, CV_8UC3, Scalar(0));
    imshow("rectangles", image);
    waitKey();
    RotatedRect originalRect;
    Point2f vertices[4];
    vector<Point2f> vertVect;
    RotatedRect calculatedRect;

    while (waitKey(0) != 27) {
        // Create a rectangle, rotating it by 10 degrees more each time.
        originalRect = RotatedRect(Point2f(150, 150), Size2f(200, 100), angle);

        // Convert the rectangle to a vector of points for minAreaRect to use.
        // Also move the points to the right, so that the two rectangles aren't
        // in the same place.
        originalRect.points(vertices);
        for (int i = 0; i < 4; i++) {
            vertVect.push_back(vertices[i]);
        }

        // Get minAreaRect to find a rectangle that encloses the points. This
        // should have the exact same orientation as our original rectangle.
        calculatedRect = minAreaRect(vertVect);

        // Draw the original rectangle, and the one given by minAreaRect.
        for (int i = 0; i < 4; i++) {
            // line(image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
            line(image, vertVect[i], vertVect[(i + 1) % 4], Scalar(0, 0, 255));
        }
        float newangle = calculatedRect.angle;
        if (calculatedRect.size.width < calculatedRect.size.height) {
            newangle = calculatedRect.angle - 90;
        }
        imshow("rectangles", image);

        system("cls");
        printf("Angle given by minAreaRect: %7.2f\n", calculatedRect.angle);
        printf("New angle:                  %7.2f\n", newangle);
        printf("w = %d \n", (int)round(calculatedRect.size.width));
        printf("h = %d \n", (int)round(calculatedRect.size.height));

        // Reset everything for the next frame.
        image = Mat(400, 400, CV_8UC3, Scalar(0));
        vertVect.clear();
        angle += 10;
    }

    return 0;
}