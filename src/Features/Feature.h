#ifndef FEATURE_H
#define FEATURE_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

struct FeaturePoint{
  cv::Point2f  point;
  int id;
  int age;
};

struct FeatureSet {
    std::vector<cv::Point2f>  points;
    std::vector<int>  ages;
    int size(){
        return points.size();
    }
    void clear(){
        points.clear();
        ages.clear();
    }
 };

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points);

void opticalFlow(const cv::Mat &img_l_0, const cv::Mat &img_l_1, std::vector<cv::Point2f> *points_l_0,
                 std::vector<cv::Point2f> *points_l_1, std::vector<uchar>* status);

#if USE_CUDA
  void circularMatching_gpu(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                        std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                        std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                        std::vector<cv::Point2f>& points_l_0_return,
                        FeatureSet& current_features);
#endif


void appendNewFeatures(const cv::Mat &image, FeatureSet *current_features);

#endif
