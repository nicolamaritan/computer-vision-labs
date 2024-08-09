#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

const char* ROBOCUP_WINDOW_NAME = "Robocup";

inline bool is_valid(Mat& image, int row, int col);
Vec3b mean_9(Mat& image, int row, int col);
void mean(int event, int x, int y, int flags, void* userdata);
void segment(int event, int x, int y, int flags, void* userdata);
void mask(Mat& src, Mat& dst, Vec3b reference, unsigned int T);
void apply_mask(Mat& src, Mat& dst, Vec3b reference);
bool is_within_threshold(Vec3b vec, Vec3b reference, unsigned int T);


int main(int argc, char** argv)
{
    Mat robocup = imread("robocup.jpg");
    Mat robocup_HSV = Mat(robocup.rows, robocup.cols, CV_8UC3);
    cvtColor(robocup, robocup_HSV, COLOR_BGR2HSV);
    namedWindow(ROBOCUP_WINDOW_NAME);
    setMouseCallback(ROBOCUP_WINDOW_NAME, segment, &robocup_HSV);
    imshow(ROBOCUP_WINDOW_NAME, robocup_HSV);
    waitKey(0);

    return 0;
}

inline bool is_valid(Mat& image, int row, int col)
{
    return col < image.cols && col >= 0 && row < image.rows && row >= 0;
}

Vec3b mean_9(Mat& image, int row, int col)
{
    const int SIZE = 9;
    int counted = 0;
    Vec<unsigned int, 3> means;

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

void mean(int event, int x, int y, int flags, void* userdata)
{
    Mat* image = reinterpret_cast<Mat*>(userdata);
    if  ( event == EVENT_LBUTTONDOWN )
    {
        std::cout << mean_9(*image, y, x) << std::endl;
    }
}

void segment(int event, int x, int y, int flags, void* userdata)
{
    Mat image = *(Mat*)userdata;
    if  ( event == EVENT_LBUTTONDOWN )
    {
        Vec3b reference_color = mean_9(image, y, x);

        Mat mask(image.rows, image.cols, CV_8U);
        apply_mask(image, mask, reference_color);
        imshow(ROBOCUP_WINDOW_NAME, mask);
        waitKey(0);

    }
}

void apply_mask(Mat& src, Mat& dst, Vec3b reference)
{
    const int T = 90;
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