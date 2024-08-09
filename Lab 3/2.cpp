#include <iostream>
#include <opencv2/highgui.hpp>

const std::string ROBOCUP_WINDOW_NAME = "Robocup";

void printBGR(int event, int x, int y, int flags, void* userdata)
{
    cv::Mat* image = reinterpret_cast<cv::Mat*>(userdata);
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << image->at<cv::Vec3b>(y, x) << std::endl;
    }
}

int main(int argc, char** argv)
{
    cv::Mat robocup = cv::imread("robocup.jpg");
    cv::namedWindow(ROBOCUP_WINDOW_NAME);
    cv::setMouseCallback(ROBOCUP_WINDOW_NAME, printBGR, &robocup);
    cv::imshow(ROBOCUP_WINDOW_NAME, robocup);
    cv::waitKey(0);

    return 0;
}

