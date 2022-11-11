//
// Created by magnus on 11/2/22.
//

#ifndef MULTISENSE_VIEWER_GRAPH_SLAM_H
#define MULTISENSE_VIEWER_GRAPH_SLAM_H

#include "string"
#include "MultiSense/Src/Tools/Utils.h"
#include "filesystem"
#include "opencv2/opencv.hpp"

namespace GSlam {

    static void parseAndSortFileNames(std::vector<std::string> *leftFilesNames,
                                      std::vector<std::string> *rightFilesNames,
                                      std::vector<std::string> *depthFileNames) {
        std::string rightPath = Utils::getAssetsPath() + "../../Slam/B7/B-7_img_rect_right/";
        std::string leftPath = Utils::getAssetsPath() + "../../Slam/B7/B-7_img_rect_left/";
        std::string depthPath = Utils::getAssetsPath() + "../../Slam/B7/B-7_img_rect_depth/";

        leftFilesNames->reserve(7000);
        rightFilesNames->reserve(7000);
        depthFileNames->reserve(7000);

        for (const auto &entry: std::filesystem::directory_iterator(leftPath))
            leftFilesNames->push_back(entry.path());
        for (const auto &entry: std::filesystem::directory_iterator(rightPath))
            rightFilesNames->push_back(entry.path());
        for (const auto &entry: std::filesystem::directory_iterator(depthPath))
            depthFileNames->push_back(entry.path());

        std::sort(leftFilesNames->begin(), leftFilesNames->end());
        std::sort(rightFilesNames->begin(), rightFilesNames->end());
        std::sort(depthFileNames->begin(), depthFileNames->end());

    }

    struct FeatureSet {
        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
    };

    struct FeatureMatches {
        std::vector<cv::DMatch> match1;
        std::vector<cv::DMatch> match2;
    };


    static FeatureSet getFeatures(const cv::Mat &image) {
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        orb->detectAndCompute(image, cv::Mat(), keyPoints, descriptors);

        return {keyPoints, descriptors};
    }

    static FeatureMatches featureMatcher(std::queue<FeatureSet> featureQueue, size_t id) {
        // Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch> > knn_matches;
        GSlam::FeatureSet featureSet_t0 = featureQueue.front();
        featureQueue.pop();
        matcher.knnMatch(featureQueue.front().descriptor, featureSet_t0.descriptor, knn_matches, 2);

        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> good_matches;
        for (auto &knn_matche: knn_matches) {
            if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
                good_matches.push_back(knn_matche[0]);
            }
        }

        return {good_matches, good_matches};
    }

    static cv::Mat stereoMatcher(const cv::Mat &left, const cv::Mat &right) {
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(-3,    //int minDisparity
                                                              128,     //int numDisparities
                                                              5,      //int SADWindowSize
                                                              60,    //int P1 = 0
                                                              2400,   //int P2 = 0
                                                              90,     //int disp12MaxDiff = 0
                                                              16,     //int preFilterCap = 0
                                                              1,      //int uniquenessRatio = 0
                                                              60,    //int speckleWindowSize = 0
                                                              20,     //int speckleRange = 0
                                                              cv::StereoSGBM::MODE_HH);  //bool fullDP = false
        cv::Mat disparity;

        sgbm->compute(left, right, disparity);
        return disparity;
    }
}


#endif //MULTISENSE_VIEWER_GRAPH_SLAM_H
