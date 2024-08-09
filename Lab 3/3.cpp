#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;

const std::string ROBOCUP_WINDOW_NAME = "Robocup";

inline bool is_valid(cv::Mat& image, int row, int col);
cv::Vec3b mean_9(cv::Mat& image, int row, int col);
void printBGR(int event, int x, int y, int flags, void* userdata);


int main(int argc, char** argv)
{
    cv::Mat robocup = cv::imread("robocup.jpg");
    cv::namedWindow(ROBOCUP_WINDOW_NAME);
    cv::setMouseCallback(ROBOCUP_WINDOW_NAME, printBGR, &robocup);
    cv::imshow(ROBOCUP_WINDOW_NAME, robocup);
    cv::waitKey(0);

    return 0;
}

inline bool is_valid(cv::Mat& image, int row, int col)
{
    return col < image.cols && col >= 0 && row < image.rows && row >= 0;
}

cv::Vec3b mean_9(cv::Mat& image, int row, int col)
{
    const int SIZE = 9;
    int counted = 0;
    cv::Vec<unsigned int, 3> means;

    for (int filter_row = row - SIZE/2; filter_row <= row + SIZE/2; filter_row++)
    {
        for (int filter_col = col - SIZE/2; filter_col <= col + SIZE/2; filter_col++)
        {
            if (is_valid(image, filter_row, filter_col))
            {
                counted++;
                Vec3b pixel = image.at<Vec3b>(filter_row, filter_col);
                for (int i = 0; i < 3; i++)
                {
                    means[i] += pixel[i];
                }
            }
        }
    }

    // Compute means
    for (int i = 0; i < 3; i++)
    {
        means[i] /= counted;
    }

    return means;
    
}

void printBGR(int event, int x, int y, int flags, void* userdata)
{
    cv::Mat* image = (cv::Mat*)userdata;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << mean_9(*image, y, x) << std::endl;
    }
}