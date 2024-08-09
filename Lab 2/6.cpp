#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <assert.h>

const char* GARDEN_WINDOW_NAME = "Garden";
const char* GARDEN_GRAYSCALE_WINDOW_NAME = "Grayscaled Garden";
const char* GARDEN_GRAYSCALE_HISTOGRAM_WINDOW_NAME = "Garden - Histogram";
const char* GARDEN_EQUALIZED_WINDOW_NAME = "Garden - Equalized";
const char* GARDEN_GRAYSCALE_EQUALIZED_HISTOGRAM_WINDOW_NAME = "Garden - Equalized Histogram";

using namespace cv;

int main(int argc, char** argv)
{
    // ---------------- Grayscale garden ----------------
    Mat garden = imread("Images/Garden.jpg");
    namedWindow(GARDEN_WINDOW_NAME);
    imshow(GARDEN_WINDOW_NAME, garden);
    waitKey(0);

    Mat garden_grayscale(garden.rows, garden.cols, CV_8U);
    cvtColor(garden, garden_grayscale, COLOR_BGR2GRAY);
    assert(garden_grayscale.channels() == 1);


    namedWindow(GARDEN_GRAYSCALE_WINDOW_NAME);
    imshow(GARDEN_GRAYSCALE_WINDOW_NAME, garden_grayscale);
    waitKey(0);

    imwrite("Images/Garden_grayscale.jpg", garden_grayscale);

    // ---------------- Histogram ----------------
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    Mat histogram;
    calcHist(&garden_grayscale, 1, 0, Mat(), histogram, 1, &histSize, histRange, true, false);

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

    // Normalize values for visualization
    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for( int i = 1; i < histSize; i++ )
    {
        line(histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram.at<float>(i-1) ) ),
            Point( bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
            Scalar( 255, 255, 255), 2, 8, 0);
    }
    imshow(GARDEN_GRAYSCALE_HISTOGRAM_WINDOW_NAME, histImage);
    imwrite("Images/Garden_histogram.jpg", histImage);
    waitKey(0);

    // ---------------- Histogram Equalization ----------------
    Mat equalized_garden;
    equalizeHist(garden_grayscale, equalized_garden);
    Mat equalized_histogram;
    calcHist(&equalized_garden, 1, 0, Mat(), equalized_histogram, 1, &histSize, histRange, true, false);

    Mat equalizedHistImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

    // Normalize values for visualization
    normalize(equalized_histogram, equalized_histogram, 0, equalizedHistImage.rows, NORM_MINMAX, -1, Mat());

    for( int i = 1; i < histSize; i++ )
    {
        line(equalizedHistImage, Point( bin_w*(i-1), hist_h - cvRound(equalized_histogram.at<float>(i-1) ) ),
            Point( bin_w*(i), hist_h - cvRound(equalized_histogram.at<float>(i))),
            Scalar( 255, 255, 255), 2, 8, 0);
    }

    namedWindow(GARDEN_EQUALIZED_WINDOW_NAME);
    imshow(GARDEN_EQUALIZED_WINDOW_NAME, equalized_garden);
    waitKey(0);

    imshow(GARDEN_GRAYSCALE_EQUALIZED_HISTOGRAM_WINDOW_NAME, equalizedHistImage);
    imwrite("Images/Garden_equalized_histogram.jpg", equalizedHistImage);
    waitKey(0);
}
