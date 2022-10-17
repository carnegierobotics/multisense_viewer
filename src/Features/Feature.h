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

class Bucket
{

public:
    int id;
    int max_size;

    FeatureSet features;

    Bucket(int size){
        max_size = size;
    }

    int size(){
        return features.points.size();
    }


    void add_feature(cv::Point2f point, int age){
        // won't add feature with age > 10;
        int age_threshold = 10;
        if (age < age_threshold)
        {
            // insert any feature before bucket is full
            if (size()<max_size)
            {
                features.points.push_back(point);
                features.ages.push_back(age);

            }
            else
                // insert feature with old age and remove youngest one
            {
                int age_min = features.ages[0];
                int age_min_idx = 0;

                for (int i = 0; i < size(); i++)
                {
                    if (age < age_min)
                    {
                        age_min = age;
                        age_min_idx = i;
                    }
                }
                features.points[age_min_idx] = point;
                features.ages[age_min_idx] = age;
            }
        }

    }

    void get_features(FeatureSet& current_features){

        current_features.points.insert(current_features.points.end(), features.points.begin(), features.points.end());
        current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());
    }

    ~Bucket(){
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
void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                                 std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                                 std::vector<cv::Point2f>& points0_return,
                                 std::vector<uchar>& status0, std::vector<uchar>& status1,
                                 std::vector<uchar>& status2, std::vector<uchar>& status3,
                                 std::vector<int>& ages);


void
checkValidMatch(std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &points_return, std::vector<bool> &status,
                int threshold);

void removeInvalidPoints(std::vector<cv::Point2f> &points, const std::vector<bool> &status);

void trackingFrame2Frame(cv::Mat &projMatrl, cv::Mat &projMatrr,
                         std::vector<cv::Point2f> &pointsLeft_t0,
                         std::vector<cv::Point2f> &pointsLeft_t1,
                         cv::Mat &points3D_t0,
                         cv::Mat &rotation,
                         cv::Mat &translation,
                         bool mono_rotation);




void displayTracking(cv::Mat &imageLeft_t1,
                     std::vector<cv::Point2f> &pointsLeft_t0,
                     std::vector<cv::Point2f> &pointsLeft_t1);

void integrateOdometryStereo(int frame_i, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, const cv::Mat& translation_stereo);
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose);

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket);
#endif