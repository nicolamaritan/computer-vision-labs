
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

const std::string CIRCLE_MAP = "Line Map";

int main(int argc, char **argv)
{
    Mat street, street_gray, edges;

    street = imread("street_scene.png");
    cvtColor(street, street_gray, COLOR_BGR2GRAY);
    blur(street_gray, street_gray, Size(3, 3));
    namedWindow(CIRCLE_MAP, WINDOW_AUTOSIZE);

    // Standard Hough Line Transform
    vector<Vec3f> circles; // will hold the results of the detection
    HoughCircles(street_gray, circles, HOUGH_GRADIENT, 1,
                 street_gray.cols / 1,
                 255, 0.99, 7, 7

    );

    cout << circles.size();

    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle(street, center, 1, Scalar(0, 100, 100), 2, LINE_AA);
        // circle outline
        int radius = c[2];
        circle(street, center, radius, Scalar(255, 0, 255), 2, LINE_AA);
    }

    imshow(CIRCLE_MAP, street);
    waitKey(0);
    return 0;
}