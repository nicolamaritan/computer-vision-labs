#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <assert.h>

using namespace cv;
using namespace std;

string FEATURE_MATCHING_WINDOW = "Feature matching window";

void equalizeYUV(const Mat &src, Mat &dst);
bool is_nice_homography(const Mat &H);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Wrong input.\n";
        return 1;
    }

    Mat input1 = imread(argv[1]);
    Mat input2 = imread(argv[2]);

    Ptr<SIFT> detector = SiftFeatureDetector::create(0, 3, 0.08, 10.0, 2);
    std::vector<KeyPoint> keypoints1;
    Mat descriptors1;
    detector->detect(input1, keypoints1);
    detector->compute(input1, keypoints1, descriptors1);

    std::vector<KeyPoint> keypoints2;
    Mat descriptors2;
    detector->detect(input2, keypoints2);
    detector->compute(input2, keypoints2, descriptors2);

    Mat output1;
    // drawKeypoints(input1, keypoints1, output1);
    // imshow(FEATURE_MATCHING_WINDOW, output1);
    // waitKey(0);

    Mat output2;
    // drawKeypoints(input2, keypoints2, output2);
    // imshow(FEATURE_MATCHING_WINDOW, output2);
    // waitKey(0);

    vector<vector<DMatch>> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
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

    double ratio = (double)good_matches.size() / matches.size();
    cout << "matches size: " << matches.size() << endl;
    cout << "good_matches size: " << good_matches.size() << endl;
    cout << "ratio: " << ratio << endl;

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
    Mat H = findHomography(input1_good_matches_points, input2_good_matches_points, RANSAC, 3.0, mask);
    cout << "non zeroes: " << countNonZero(mask) << endl;

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> input1_corners(4);
    input1_corners[0] = Point2f(0, 0);
    input1_corners[1] = Point2f(input1.cols, 0);
    input1_corners[2] = Point2f(input1.cols, input1.rows);
    input1_corners[3] = Point2f(0, input1.rows);
    std::vector<Point2f> input2_corners(4);

    perspectiveTransform(input1_corners, input2_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, input2_corners[0] + Point2f(input1.cols, 0), input2_corners[1] + Point2f(input1.cols, 0), Scalar(0, 255, 0), 10);
    line(img_matches, input2_corners[1] + Point2f(input1.cols, 0), input2_corners[2] + Point2f(input1.cols, 0), Scalar(0, 255, 0), 10);
    line(img_matches, input2_corners[2] + Point2f(input1.cols, 0), input2_corners[3] + Point2f(input1.cols, 0), Scalar(0, 255, 0), 10);
    line(img_matches, input2_corners[3] + Point2f(input1.cols, 0), input2_corners[0] + Point2f(input1.cols, 0), Scalar(0, 255, 0), 10);

    if (is_nice_homography(H))
    {
        cout << "The two images have SIMILAR content." << endl;
    }
    else
    {
        cout << "The two images have DIFFERENT content." << endl;
    }

    //-- Show detected matches
    imshow(FEATURE_MATCHING_WINDOW, img_matches);
    waitKey(0);

    return 0;
}

void equalizeYUV(const Mat &src, Mat &dst)
{
    cvtColor(src, dst, COLOR_BGR2YUV);
    vector<Mat> channels(3);
    split(dst, channels);
    equalizeHist(channels[0].clone(), channels[0]);
    merge(channels, dst);
    cvtColor(dst.clone(), dst, COLOR_YUV2BGR);
}

bool is_nice_homography(const Mat &H)
{
    /*
        From https://github.com/MasteringOpenCV/code/issues/11#issuecomment-20632826

        So in the first check we compute the determinant of the 2x2 submatrix of homography matrix.
        This [2x2] matrix called R contains rotation component
        of the estimated transformation. Correct rotation matrix has it's determinant value equals to 1.
        In our case R matrix may contain scale component, so it's determinant can have other values,
        but in general for correct rotation and scale values it's always greater than zero.
    */
    const double det = H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(1, 0) * H.at<double>(0, 1);
    if (det < 0)
        return false;

    /*
        To understand other checks i write the simplified form of homography matrix:
            ( s * cos(t), -sin(t),    tx)
            (     sin(t), s * cos(t), ty)
            ( px,         py,         1)
        It's an approximation of the homography matrix, but in general the top-left [2x2]
        matrix represents scale and rotation, tx and ty - translation and px, py - projective transformation.
        These thresholds were choosen empirically.
    */

    const double N1 = sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0));
    if (N1 > 4 || N1 < 0.1)
        return false;

    const double N2 = sqrt(H.at<double>(0, 1) * H.at<double>(0, 1) + H.at<double>(1, 1) * H.at<double>(1, 1));
    if (N2 > 4 || N2 < 0.1)
        return false;

    const double N3 = sqrt(H.at<double>(2, 0) * H.at<double>(2, 0) + H.at<double>(2, 1) * H.at<double>(2, 1));
    if (N3 > 0.002)
        return false;

    return true;
}