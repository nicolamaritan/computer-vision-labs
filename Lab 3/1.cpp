#include <iostream>
#include <opencv2/highgui.hpp>

const std::string ROBOCUP_WINDOW_NAME = "Robocup";

int main(int argc, char** argv)
{
    cv::Mat robocup = cv::imread("robocup.jpg");
    cv::namedWindow(ROBOCUP_WINDOW_NAME);
    cv::imshow(ROBOCUP_WINDOW_NAME, robocup);
    cv::waitKey(0);

    return 0;
}

