
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
Mat street, street_gray;
Mat dst, detected_edges;
int threshold_1 = 0;
int threshold_2 = 0;
const int max_threshold_1 = 500;
const int max_threshold_2 = 500;
const int ratio = 3;
const int kernel_size = 3;
const std::string EDGE_MAP = "Edge Map";

static void CannyDetector(int, void *)
{
    blur(street_gray, detected_edges, Size(3, 3));
    Canny(detected_edges, detected_edges, threshold_1, threshold_2, kernel_size);
    dst = Scalar::all(0);
    street.copyTo(dst, detected_edges);
    imshow(EDGE_MAP, dst);
}

int main(int argc, char **argv)
{
    street = imread("street_scene.png");
    dst.create(street.size(), street.type());
    cvtColor(street, street_gray, COLOR_BGR2GRAY);
    namedWindow(EDGE_MAP, WINDOW_AUTOSIZE);
    createTrackbar("Threshold 1:", EDGE_MAP, &threshold_1, max_threshold_1, CannyDetector);
    createTrackbar("Threshold 2:", EDGE_MAP, &threshold_2, max_threshold_2, CannyDetector);
    CannyDetector(0, 0);
    waitKey(0);
    return 0;
}