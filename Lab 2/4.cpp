#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <assert.h>
#include "ImageFilters.h"

const char* GARDEN_WINDOW_NAME = "Garden";
const char* GARDEN_GRAYSCALE_WINDOW_NAME = "Grayscaled Garden";
const char* LENA_MAX_FILTER_WINDOW_NAME = "Lena - Max filter";
const char* LENA_MIN_FILTER_WINDOW_NAME = "Lena - Min filter";
const char* ASTRONAUT_MAX_FILTER_WINDOW_NAME = "Astronaut - Max filter";
const char* ASTRONAUT_MIN_FILTER_WINDOW_NAME = "Astronaut - Min filter";
const char* GARDEN_MAX_FILTER_WINDOW_NAME = "Garden - Max filter";
const char* GARDEN_MEDIAN_FILTER_WINDOW_NAME = "Garden - Median filter";
const char* LENA_MEDIAN_FILTER_WINDOW_NAME = "Lena - Median filter";
const char* ASTRONAUT_MEDIAN_FILTER_WINDOW_NAME = "Astronaut - Median filter";
const char* LENA_GAUSSIAN_BLUR_WINDOW_NAME = "Lena - Gaussian blur";
const char* ASTRONAUT_GAUSSIAN_BLUR_WINDOW_NAME = "Astronaut - Gaussian blur";
const char* GARDEN_GAUSSIAN_BLUR_WINDOW_NAME = "Garden - Gaussian blur";

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Select the filter size.\n";
        return 1;
    }
    int filter_size = atoi(argv[1]);
    // ---------------- Grayscale garden ----------------
    cv::Mat garden = cv::imread("Images/Garden.jpg");
    cv::namedWindow(GARDEN_WINDOW_NAME);
    cv::imshow(GARDEN_WINDOW_NAME, garden);
    cv::waitKey(0);

    cv::Mat garden_grayscale(garden.rows, garden.cols, CV_8U);
    cv::cvtColor(garden, garden_grayscale, cv::COLOR_BGR2GRAY);

    cv::namedWindow(GARDEN_GRAYSCALE_WINDOW_NAME);
    cv::imshow(GARDEN_GRAYSCALE_WINDOW_NAME, garden_grayscale);
    cv::waitKey(0);

    cv::imwrite("Images/Garden_grayscale.jpg", garden_grayscale);


    // ---------------- Lena corrupted ----------------
    cv::Mat lena = cv::imread("Images/Lena_corrupted.png");
    cv::Mat lena_grayscale(lena.rows, lena.cols, CV_8U);
    cv::cvtColor(lena, lena_grayscale, cv::COLOR_BGR2GRAY);

    // Max filter
    cv::Mat lena_max(lena_grayscale.rows, lena_grayscale.cols, CV_8UC1);
    max_filter(lena_grayscale, lena_max, filter_size);
    cv::namedWindow(LENA_MAX_FILTER_WINDOW_NAME);
    cv::imshow(LENA_MAX_FILTER_WINDOW_NAME, lena_max);
    cv::waitKey(0);

    // Min filter
    cv::Mat lena_min(lena_grayscale.rows, lena_grayscale.cols, CV_8UC1);
    min_filter(lena_grayscale, lena_min, filter_size);
    cv::namedWindow(LENA_MIN_FILTER_WINDOW_NAME);
    cv::imshow(LENA_MIN_FILTER_WINDOW_NAME, lena_min);
    cv::waitKey(0);

    // ---------------- Astronaut corrupted ----------------
    cv::Mat astronaut = cv::imread("Images/Astronaut_salt_pepper.png");
    cv::Mat astronaut_grayscale(astronaut.rows, astronaut.cols, CV_8U);
    cv::cvtColor(astronaut, astronaut_grayscale, cv::COLOR_BGR2GRAY);

    // Max filter
    cv::Mat astronaut_max(astronaut_grayscale.rows, astronaut_grayscale.cols, CV_8UC1);
    max_filter(astronaut_grayscale, astronaut_max, filter_size);
    cv::namedWindow(ASTRONAUT_MAX_FILTER_WINDOW_NAME);
    cv::imshow(ASTRONAUT_MAX_FILTER_WINDOW_NAME, astronaut_max);
    cv::waitKey(0);

    // Min filter
    cv::Mat astronaut_min(astronaut_grayscale.rows, astronaut_grayscale.cols, CV_8UC1);
    min_filter(astronaut_grayscale, astronaut_min, filter_size);
    cv::namedWindow(ASTRONAUT_MIN_FILTER_WINDOW_NAME);
    cv::imshow(ASTRONAUT_MIN_FILTER_WINDOW_NAME, astronaut_min);
    cv::waitKey(0);

    // ---------------- Remove black cables ----------------
    // We apply a max filter since the cables are way darker than the background.
    cv::Mat garden_max(garden_grayscale.rows, garden_grayscale.cols, CV_8UC1);
    max_filter(garden_grayscale, garden_max, filter_size);
    cv::namedWindow(GARDEN_MAX_FILTER_WINDOW_NAME);
    cv::imshow(GARDEN_MAX_FILTER_WINDOW_NAME, garden_max);
    cv::waitKey(0);

    // ---------------- Median filter ----------------
    cv::Mat lena_median(lena_grayscale.rows, lena_grayscale.cols, CV_8UC1);
    median_filter(lena_grayscale, lena_median, filter_size);
    cv::namedWindow(LENA_MEDIAN_FILTER_WINDOW_NAME);
    cv::imshow(LENA_MEDIAN_FILTER_WINDOW_NAME, lena_median);
    cv::waitKey(0);

    cv::Mat astronaut_median(astronaut_grayscale.rows, astronaut_grayscale.cols, CV_8UC1);
    median_filter(astronaut_grayscale, astronaut_median, filter_size);
    cv::namedWindow(ASTRONAUT_MEDIAN_FILTER_WINDOW_NAME);
    cv::imshow(ASTRONAUT_MEDIAN_FILTER_WINDOW_NAME, astronaut_median);
    cv::waitKey(0);

    cv::Mat garden_median(garden_grayscale.rows, garden_grayscale.cols, CV_8UC1);
    median_filter(garden_grayscale, garden_median, filter_size);
    cv::namedWindow(GARDEN_MEDIAN_FILTER_WINDOW_NAME);
    cv::imshow(GARDEN_MEDIAN_FILTER_WINDOW_NAME, garden_median);
    cv::waitKey(0);

    // ---------------- Gaussian blur ----------------
    cv::Mat lena_gaussian(lena_grayscale.rows, lena_grayscale.cols, CV_8UC1);
    GaussianBlur(lena_grayscale, lena_gaussian, cv::Size(filter_size, filter_size), 0, 0); 
    cv::namedWindow(LENA_GAUSSIAN_BLUR_WINDOW_NAME);
    cv::imshow(LENA_GAUSSIAN_BLUR_WINDOW_NAME, lena_gaussian);
    cv::waitKey(0);    

    cv::Mat astronaut_gaussian(astronaut_grayscale.rows, astronaut_grayscale.cols, CV_8UC1);
    GaussianBlur(astronaut_grayscale, astronaut_gaussian, cv::Size(filter_size, filter_size), 0, 0); 
    cv::namedWindow(ASTRONAUT_GAUSSIAN_BLUR_WINDOW_NAME);
    cv::imshow(ASTRONAUT_GAUSSIAN_BLUR_WINDOW_NAME, astronaut_gaussian);
    cv::waitKey(0);    

    cv::Mat garden_gaussian(garden_grayscale.rows, garden_grayscale.cols, CV_8UC1);
    GaussianBlur(garden_grayscale, garden_gaussian, cv::Size(filter_size, filter_size), 0, 0); 
    cv::namedWindow(GARDEN_GAUSSIAN_BLUR_WINDOW_NAME);
    cv::imshow(GARDEN_GAUSSIAN_BLUR_WINDOW_NAME, garden_gaussian);
    cv::waitKey(0);    


    return 0;
}
