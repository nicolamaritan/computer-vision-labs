#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "panoramic_utils.h"
#include "Stitcher.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <assert.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

const string FEATURE_MATCHING_WINDOW = "Feature matching window";
const string UNCUT_PANORAMA_WINDOW = "Uncut panorama window";
const string PANORAMA_WINDOW = "Panorama window";

const int ANGLE = 33;
const int DOLOMITES_ANGLE = 27;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Wrong input.\n";
        return 1;
    }

    String directory(argv[1]);
    vector<String> filenames;
    utils::fs::glob(directory, "i*", filenames, false);
    vector<Mat> images;
    double angle = ANGLE;

    if (directory.find("dolomites") != String::npos)
    {
        angle = DOLOMITES_ANGLE;
    }
    cout << "angle: " << angle << endl;

    for (auto filename : filenames)
    {
        cout << filename << endl;
        images.push_back(cylindricalProj(imread(filename), angle));
    }

    Mat panorama;
    Stitcher stitcher = Stitcher();
    stitcher.stitch(images, panorama);

    imwrite("panorama.jpeg", panorama);

    return 0;
}