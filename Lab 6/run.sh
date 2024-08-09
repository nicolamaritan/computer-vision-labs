#!/bin/bash

clear
g++ 1.cpp -I /usr/include/opencv4 -l opencv_highgui -l opencv_imgproc -l opencv_core -l opencv_imgcodecs -l opencv_features2d

# Iterate through each pair of images
for image1 in Images/* 
do
    for image2 in Images/* 
    do 
        echo "./a.out $image1 $image2"
        # Run a.out with the current pair of images
        ./a.out "$image1" "$image2"
        
    done
done
