#ifndef IMAGE_FILTERS_H
#define IMAGE_FILTERS_H
#include <iostream>
#include <opencv2/highgui.hpp>

void max_filter(cv::Mat& src, cv::Mat& dst, int size);
void min_filter(cv::Mat& src, cv::Mat& dst, int size);
void median_filter(cv::Mat& src, cv::Mat& dst, int size);

#endif
