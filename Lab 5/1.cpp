#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <assert.h>

using namespace cv;
using namespace std;

const std::string ASPHALT_WINDOW = "Asphalt window.";

void contrast_stretch(const Mat &src, Mat &dst, float low_level_ratio);
void segment(Mat &src, Mat &dst);
void show_asphalt(Mat &img);

int main(int argc, char **argv)
{
    Mat asphalt_1 = imread("Asphalt cracks/Asphalt-1.png");
    Mat asphalt_2 = imread("Asphalt cracks/Asphalt-2.png");
    Mat asphalt_3 = imread("Asphalt cracks/Asphalt-3.png");
    Mat asphalt_1_gray;
    Mat asphalt_2_gray;
    Mat asphalt_3_gray;

    cvtColor(asphalt_1, asphalt_1_gray, COLOR_BGR2GRAY);
    cvtColor(asphalt_2, asphalt_2_gray, COLOR_BGR2GRAY);
    cvtColor(asphalt_3, asphalt_3_gray, COLOR_BGR2GRAY);
    namedWindow(ASPHALT_WINDOW, WINDOW_AUTOSIZE);

    Mat dst;
    segment(asphalt_1_gray, dst);
    segment(asphalt_2_gray, dst);
    segment(asphalt_3_gray, dst);
}

void segment(Mat &src, Mat &dst)
{
    equalizeHist(src, dst);
    //show_asphalt(dst);

    contrast_stretch(dst.clone(), dst, 8);
    //show_asphalt(dst);

    bilateralFilter(dst.clone(), dst, 25, 20, 230, BORDER_DEFAULT);
    //show_asphalt(dst);

    medianBlur(dst.clone(), dst, 3);
    //show_asphalt(dst);

    bilateralFilter(dst.clone(), dst, 31, 40, 130, BORDER_DEFAULT);
    //show_asphalt(dst);

    medianBlur(dst.clone(), dst, 3);
    //show_asphalt(dst);

    threshold(dst.clone(), dst, 70, 255, THRESH_BINARY_INV);
    show_asphalt(dst);
}

void show_asphalt(Mat &img)
{
    imshow(ASPHALT_WINDOW, img);
    waitKey(0);
}

void contrast_stretch(const Mat &src, Mat &dst, float low_level_ratio)
{
    uchar ref = 235;
    Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++)
    {
        if (i < ref / low_level_ratio)
        {
            lut.at<uchar>(0, i) = saturate_cast<uchar>((uchar)(low_level_ratio * i));
        }
        else
        {
            uchar m = (255 - ref) / (255 - ref / low_level_ratio);
            uchar q = ref - m * (ref / low_level_ratio);
            lut.at<uchar>(0, i) = saturate_cast<uchar>((uchar)(m * i + q));
        }
    }

    // Apply the gamma correction using the lookup table
    LUT(src, lut, dst);
}