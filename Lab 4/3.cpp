
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

const std::string LINE_MAP = "Line Map";

cv::Point intersection(cv::Point p1, cv::Point p2, cv::Point p3, cv::Point p4)
{
    // Funny formula: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    double x1 = p1.x, y1 = p1.y;
    double x2 = p2.x, y2 = p2.y;
    double x3 = p3.x, y3 = p3.y;
    double x4 = p4.x, y4 = p4.y;

    double denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4));

    double px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
    double py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

    return cv::Point(px, py);
}

int main(int argc, char **argv)
{
    Mat street, street_gray, edges;

    street = imread("street_scene.png");
    cvtColor(street, street_gray, COLOR_BGR2GRAY);
    blur(street_gray, street_gray, Size(3, 3));
    namedWindow(LINE_MAP, WINDOW_AUTOSIZE);
    Canny(street_gray, edges, 230, 250, 3);

    // Standard Hough Line Transform
    vector<Vec2f> lines;                                 // will hold the results of the detection
    HoughLines(edges, lines, 1, CV_PI / 180, 110, 0, 0); // runs the actual detection

    // Sort lines
    std::sort(lines.begin(), lines.end(),
              [](const cv::Vec2f &a, const cv::Vec2f &b)
              {
                  return a[0] > b[0];
              });

    float theta_0 = lines[0][1];
    float m_0 = -(cos(theta_0) / sin(theta_0));

    /*
        Filter the lines with:
        - same angular coefficient sign of the first (stronger) line
        - angular coefficient too similar to the one of the first (stronger) line
    */  
    int initial_lines_size = lines.size();
    for (int i = 0; i < initial_lines_size - 1; i++)
    {
        float theta_1 = lines[1][1];
        float m_1 = -(cos(theta_1) / sin(theta_1));

        if (m_1 * m_0 > 0 || abs(m_0 - m_1) < 1.8)
        {
            lines.erase(lines.begin() + 1);
        }
    }

    std::vector<std::vector<cv::Point>> points;

    // Draw the lines
    for (size_t i = 0; i < 2; i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        // line(street, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
        points.push_back({pt1, pt2});
    }

    Point lines_intersection = intersection(points[0][0], points[0][1], points[1][0], points[1][1]);
    Point bottom_left_corner(0, street.rows - 1);
    Point bottom_right_corner(street.cols - 1, street.rows - 1);

    Point bottom_intersection_1 = intersection(lines_intersection, points[0][0], bottom_left_corner, bottom_right_corner);
    Point bottom_intersection_2 = intersection(lines_intersection, points[1][0], bottom_left_corner, bottom_right_corner);

    // Fill the area between the lines in the mask image
    fillPoly(street, std::vector({lines_intersection, bottom_intersection_1, bottom_intersection_2}), Vec3b(0, 0, 255));

    imshow(LINE_MAP, street);
    waitKey(0);
    return 0;
}