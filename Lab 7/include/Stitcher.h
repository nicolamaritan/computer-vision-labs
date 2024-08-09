#ifndef STITCHER_H
#define STITCHER_H

#include <vector>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

class Stitcher
{
public:
    void stitch(const std::vector<cv::Mat> images, cv::Mat &panorama);
};

#endif