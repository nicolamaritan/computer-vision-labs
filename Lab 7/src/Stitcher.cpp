#include "Stitcher.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <assert.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

void Stitcher::stitch(const std::vector<cv::Mat> images, cv::Mat &panorama)
{
    Ptr<SIFT> detector = SiftFeatureDetector::create(0, 3, 0.08, 10.0, 2);
    vector<Mat> homographies_only_tx;
    for (int i = 0; i < images.size() - 1; i++)
    {
        Mat input1 = images[i];
        Mat input2 = images[i + 1];

        std::vector<KeyPoint> keypoints1;
        Mat descriptors1;
        detector->detect(input1, keypoints1);
        detector->compute(input1, keypoints1, descriptors1);

        std::vector<KeyPoint> keypoints2;
        Mat descriptors2;
        detector->detect(input2, keypoints2);
        detector->compute(input2, keypoints2, descriptors2);

        Mat output1;
        Mat output2;

        vector<vector<DMatch>> matches;
        Ptr<BFMatcher> matcher = BFMatcher::create();
        matcher->knnMatch(descriptors1, descriptors2, matches, 2);

        // Mat image_matches;

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }

        //-- Draw matches
        Mat img_matches;
        drawMatches(input1, keypoints1, input2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // imshow(FEATURE_MATCHING_WINDOW, img_matches);
        // waitKey(0);

        //-- Localize the object
        std::vector<Point2f> input1_good_matches_points;
        std::vector<Point2f> input2_good_matches_points;

        for (int i = 0; i < good_matches.size(); i++)
        {
            //-- Get the keypoints from the good matches
            input1_good_matches_points.push_back(keypoints1[good_matches[i].queryIdx].pt);
            input2_good_matches_points.push_back(keypoints2[good_matches[i].trainIdx].pt);
        }

        Mat mask;
        Mat H = findHomography(input2_good_matches_points, input1_good_matches_points, RANSAC, 3.0, mask);
        cout << "non zeroes: " << countNonZero(mask) << endl;

        //-- Get the corners from the image_2 ( the object to be "detected" )
        std::vector<Point2f> input2_corners(4);
        input2_corners[0] = Point2f(0, 0);
        input2_corners[1] = Point2f(input2.cols, 0);
        input2_corners[2] = Point2f(input2.cols, input2.rows);
        input2_corners[3] = Point2f(0, input2.rows);
        std::vector<Point2f> input1_corners(4);

        perspectiveTransform(input2_corners, input1_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line(img_matches, input1_corners[0], input1_corners[1], Scalar(0, 255, 0), 5);
        line(img_matches, input1_corners[1], input1_corners[2], Scalar(0, 255, 0), 5);
        line(img_matches, input1_corners[2], input1_corners[3], Scalar(0, 255, 0), 5);
        line(img_matches, input1_corners[3], input1_corners[0], Scalar(0, 255, 0), 5);

        //-- Show detected matches
        // imshow(FEATURE_MATCHING_WINDOW, img_matches);
        // waitKey(0);

        Mat H_only_tx = Mat::eye(3, 3, CV_64F);
        H_only_tx.at<double>(0, 2) = H.at<double>(0, 2);
        homographies_only_tx.push_back(H_only_tx);
    }

    panorama = images[0];
    for (int i = 0; i < images.size() - 1; i++)
    {
        Mat total_homography = Mat::eye(3, 3, CV_64F);
        for (int j = 0; j <= i; j++)
        {
            total_homography.at<double>(0, 2) += homographies_only_tx[j].at<double>(0, 2);
        }

        // Point of cut is the # of cols of the current panorama plus the shift of the last homography
        int cut_x_point = panorama.cols + homographies_only_tx[i].at<double>(0, 2);
        // cout << total_homography << endl << "cut point: " << cut_x_point << endl;

        Mat result;
        warpPerspective(images[i + 1], result, total_homography, Size(images[i + 1].cols + panorama.cols, images[i + 1].rows), 1, INTER_LINEAR, BORDER_REPLICATE);
        Mat half(result, Rect(0, 0, panorama.cols, panorama.rows));
        panorama.copyTo(half);
        panorama = result;

        // imshow(UNCUT_PANORAMA_WINDOW, panorama);
        // waitKey(0);

        Mat panorama_transposed = panorama.t();
        Mat panorama_cut = Mat(cut_x_point, panorama_transposed.cols, CV_8UC1, panorama_transposed.data);
        panorama = panorama_cut.t();

        // imshow(PANORAMA_WINDOW, panorama);
        // waitKey(0);
    }
}
