#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;

const std::string ROBOCUP_WINDOW_NAME = "Robocup";

inline bool is_valid(cv::Mat& image, int row, int col);
cv::Vec3b mean_9(cv::Mat& image, int row, int col);
void printBGR(int event, int x, int y, int flags, void* userdata);
void segment(int event, int x, int y, int flags, void* userdata);
void mask(cv::Mat& src, cv::Mat& dst, Vec3b reference, unsigned int T);
void apply_mask(cv::Mat& src, cv::Mat& dst, Vec3b reference);
void apply_mask(Mat& original, Mat& mask, Mat& dst);
bool is_within_threshold(Vec3b vec, Vec3b reference, unsigned int T);


int main(int argc, char** argv)
{
    cv::Mat robocup = cv::imread("robocup.jpg");
    cv::namedWindow(ROBOCUP_WINDOW_NAME);
    cv::setMouseCallback(ROBOCUP_WINDOW_NAME, segment, &robocup);
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
    cv::Mat* image = reinterpret_cast<cv::Mat*>(userdata);
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << mean_9(*image, y, x) << std::endl;
    }
}

void segment(int event, int x, int y, int flags, void* userdata)
{
    cv::Mat image = *(cv::Mat*)userdata;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        Vec3b reference_color = mean_9(image, y, x);

        cv::Mat mask(image.rows, image.cols, CV_8U);
        apply_mask(image, mask, reference_color);
        cv::imshow(ROBOCUP_WINDOW_NAME, mask);
        cv::waitKey(0);

        cv::Mat mask_2(image.rows, image.cols, CV_8UC3);
        apply_mask(image, mask, mask_2);
        cv::imshow(ROBOCUP_WINDOW_NAME, mask_2);
        cv::waitKey(0);

    }
}

void apply_mask(cv::Mat& src, cv::Mat& dst, Vec3b reference)
{
    const int T = 70;
    for (int row = 0; row < src.rows; row++)
    {
        for (int col = 0; col < src.cols; col++)
        {
            Vec3b pixel = src.at<Vec3b>(row, col);
            if (is_within_threshold(pixel, reference, T))
            {
                dst.at<unsigned char>(row, col) = 255;
            }
            else
            {
                dst.at<unsigned char>(row, col) = 0;   
            }
        }
    }
}

void apply_mask(Mat& original, Mat& mask, Mat& dst)
{
    for (int row = 0; row < original.rows; row++)
    {
        for (int col = 0; col < original.cols; col++)
        {
            unsigned char pixel = mask.at<unsigned char>(row, col);
            if (pixel == 255)
            {
                dst.at<Vec3b>(row, col) = Vec3b(92, 37, 201);
            }
            else
            {
                dst.at<Vec3b>(row, col) = original.at<Vec3b>(row, col);
            }
        }
    }
}

bool is_within_threshold(Vec3b vec, Vec3b reference, unsigned int T)
{
    for (int i = 0; i < 3; i++)
    {
        if (abs(vec[i] - reference[i]) > T)
        {
            return false;
        }
    }
    return true;
}