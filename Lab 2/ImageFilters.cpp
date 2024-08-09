#include "ImageFilters.h"
#include <iostream>
#include <vector>
#include <algorithm>

inline bool is_valid(cv::Mat& img, int i, int j)
{
    return j < img.cols && j >= 0 && i < img.rows && i >= 0;
}

void max_filter_function(cv::Mat& src, cv::Mat& dst, int size, int i, int j)
{
    // Compute max
    unsigned char max = src.at<unsigned char>(i,j);
    for (int filter_i = i - size/2; filter_i <= i + size/2; filter_i++)
    {
        for (int filter_j = j - size/2; filter_j <= j + size/2; filter_j++)
        {
            if (is_valid(src, filter_i, filter_j))
            {
                unsigned char current = src.at<unsigned char>(filter_i, filter_j);
                if (current > max)
                {
                    max = current;
                }
            }
        }
    }
    dst.at<unsigned char>(i,j) = max;   
}

void min_filter_function(cv::Mat& src, cv::Mat& dst, int size, int i, int j)
{
    // Compute min
    unsigned char min = src.at<unsigned char>(i,j);
    for (int filter_i = i - size/2; filter_i <= i + size/2; filter_i++)
    {
        for (int filter_j = j - size/2; filter_j <= j + size/2; filter_j++)
        {
            if (is_valid(src, filter_i, filter_j))
            {
                unsigned char current = src.at<unsigned char>(filter_i, filter_j);
                if (current < min)
                {
                    min = current;
                }
            }
        }
    }
    dst.at<unsigned char>(i,j) = min;   
}

void median_filter_function(cv::Mat& src, cv::Mat& dst, int size, int i, int j)
{
    // Compute list of values
    std::vector<unsigned char> values;
    for (int filter_i = i - size/2; filter_i <= i + size/2; filter_i++)
    {
        for (int filter_j = j - size/2; filter_j <= j + size/2; filter_j++)
        {
            if (is_valid(src, filter_i, filter_j))
            {
                values.push_back(src.at<unsigned char>(filter_i, filter_j));
            }
        }
    }
    assert(values.size() != 0);

    // Compute the median
    std::sort(values.begin(), values.end());
    size_t values_size = values.size();
    dst.at<unsigned char>(i,j) = size % 2 == 0 ? values.at(values_size/2) : (values.at((values_size-1)/2) + values.at(values_size/2))/2;
}

void min_filter(cv::Mat& src, cv::Mat& dst, int size)
{
    if (size % 2 == 0)
    {
        std::cerr << "Size must be odd.\n";
        return;
    }
    if (src.channels() != dst.channels())
    {
        std::cerr << "Number of channels must match.\n";
        return;
    }

    for (int i = 0; i < src.rows; i++)
    {   
        for (int j = 0; j < src.cols; j++)
        {
            min_filter_function(src, dst, size, i, j);
        }
    }
}



void max_filter(cv::Mat& src, cv::Mat& dst, int size)
{
    if (size % 2 == 0)
    {
        std::cerr << "Size must be odd.\n";
        return;
    }
    if (src.channels() != dst.channels())
    {
        std::cerr << "Number of channels must match.\n";
        return;
    }

    for (int i = 0; i < src.rows; i++)
    {   
        for (int j = 0; j < src.cols; j++)
        {
            max_filter_function(src, dst, size, i, j);
        }
    }
}

void median_filter(cv::Mat& src, cv::Mat& dst, int size)
{
    if (size % 2 == 0)
    {
        std::cerr << "Size must be odd.\n";
        return;
    }
    if (src.channels() != dst.channels())
    {
        std::cerr << "Number of channels must match.\n";
        return;
    }

    for (int i = 0; i < src.rows; i++)
    {   
        for (int j = 0; j < src.cols; j++)
        {
            median_filter_function(src, dst, size, i, j);
        }
    }
}