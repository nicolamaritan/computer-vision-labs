#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <assert.h>

const char* GARDEN_WINDOW_NAME = "Garden";
const char* GARDEN_GRAYSCALE_WINDOW_NAME = "Grayscaled Garden";

int main(int argc, char** argv)
{
    // ---------------- Grayscale garden ----------------
    cv::Mat garden = cv::imread("Images/Garden.jpg");
    cv::namedWindow(GARDEN_WINDOW_NAME);
    cv::imshow(GARDEN_WINDOW_NAME, garden);
    cv::waitKey(0);

    cv::Mat garden_grayscale(garden.rows, garden.cols, CV_8U);
    cv::cvtColor(garden, garden_grayscale, cv::COLOR_BGR2GRAY);
    assert(garden_grayscale.channels() == 1);

    cv::namedWindow(GARDEN_GRAYSCALE_WINDOW_NAME);
    cv::imshow(GARDEN_GRAYSCALE_WINDOW_NAME, garden_grayscale);
    cv::waitKey(0);

    cv::imwrite("Images/Garden_grayscale.jpg", garden_grayscale);
    return 0;
}
