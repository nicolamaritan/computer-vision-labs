
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;

const std::string EDGE_MAP = "Edge Map";

int main(int argc, char **argv)
{
    cv::Mat gradient_angle_degrees, gradient_magnitudes;
    Mat street, street_gray, street_gray_thresholded, street_gray_smoothed;
    Mat grad_x, grad_y;

    street = imread("street_scene.png");

    cvtColor(street, street_gray, COLOR_BGR2GRAY);
    namedWindow(EDGE_MAP, WINDOW_AUTOSIZE);

    gradient_magnitudes.create(street_gray_thresholded.size(), street_gray.type());
    
    cv::GaussianBlur(street_gray, street_gray_smoothed, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
    threshold( street_gray, street_gray_thresholded, 230, 255, THRESH_TOZERO );
    cv::Sobel(street_gray_thresholded, grad_y, CV_32F, 0, 1, 3, 1, -1, cv::BORDER_DEFAULT);
    cv::Sobel(street_gray_thresholded, grad_x, CV_32F, 1, 0, 3, 1, -1, cv::BORDER_DEFAULT);
    imshow( EDGE_MAP, grad_x + grad_y );
    waitKey(0);
}