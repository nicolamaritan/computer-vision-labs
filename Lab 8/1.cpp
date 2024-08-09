#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include <assert.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

const string CHESSBOARD_CORNERS_WINDOW = "Chessboard corners window";
const string REMAPPED_WINDOW = "Remapped window";

int main(int argc, char **argv)
{
    const string directory = "./data_lab_calibration/data/checkerboard_images";
    vector<String> filenames;
    utils::fs::glob(directory, "*.png", filenames, false);
    vector<Mat> images;
    vector<Mat> images_gray;

    for (auto filename : filenames)
    {
        cout << filename << endl;
        Mat image = imread(filename);
        Mat image_grayscale;
        cvtColor(image, image_grayscale, COLOR_BGR2GRAY);

        images.push_back(image);
        images_gray.push_back(image_grayscale);
    }

    Size pattern_size(5, 6); // interior number of corners
    vector<vector<Point2f>> corners(images.size());

    for (int i = 0; i < images.size(); i++)
    {
        // Mat image_gray;
        // cvtColor(images[i], image_gray, COLOR_BGR2GRAY);

        // CALIB_CB_FAST_CHECK saves a lot of time on images
        // that do not contain any chessboard corners
        bool cornersFound = findChessboardCorners(images_gray[i], pattern_size, corners[i]);

        if (!cornersFound)
        {
            cerr << "Corners not found." << endl;
            return 1;
        }

        cornerSubPix(images_gray[i], corners[i], Size(11, 11), Size(-1, -1),
                     TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

        // drawChessboardCorners(images[i], pattern_size, Mat(corners[i]), cornersFound);
        // imshow(CHESSBOARD_CORNERS_WINDOW, images[i]);
        // waitKey(0);
    }

    vector<vector<Vec3f>> object_points;
    vector<Vec3f> single_image_points;
    for (int i = 0; i < pattern_size.height; i++)
    {
        for (int j = 0; j < pattern_size.width; j++)
        {
            single_image_points.push_back(Vec3f(j, i, 0));
        }
    }
    for (int i = 0; i < images.size(); i++)
    {
        object_points.push_back(single_image_points);
    }

    cv::Mat camera_matrix, dist_coeffs, R, T;
    double rms = calibrateCamera(object_points, corners, images[0].size(), camera_matrix, dist_coeffs, R, T);

    cout << "RMS: " << rms << endl;
    cout << camera_matrix << endl
         << R << endl;

    Mat output_map_1, output_map_2;
    initUndistortRectifyMap(camera_matrix, dist_coeffs, Mat(), Mat(), images[0].size(), CV_32FC1, output_map_1, output_map_2);
    for (auto image : images)
    {
        Mat remapped_image;
        Mat side_by_side_images;
        remap(image, remapped_image, output_map_1, output_map_2, InterpolationFlags::INTER_LINEAR);
        hconcat(image, remapped_image, side_by_side_images);
        imshow(REMAPPED_WINDOW, side_by_side_images);
        waitKey(0);
    }
    return 0;
}