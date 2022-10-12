//
// Created by magnus on 10/11/22.
//

#ifndef MULTISENSE_VIEWER_VISUALODOMETRY_H
#define MULTISENSE_VIEWER_VISUALODOMETRY_H

#include "opencv4/opencv2/opencv.hpp"
#include "MultiSense/src/Core/Definitions.h"
#include "Feature.h"

class VisualOdometry {

public:
    VisualOdometry(){

        // Create some random colors
        cv::RNG rng;
        for(int i = 0; i < 100; i++)
        {
            int r = rng.uniform(0, 256);
            int g = rng.uniform(0, 256);
            int b = rng.uniform(0, 256);
            colors.emplace_back(r,g,b);
        }

    };

    void update(VkRender::TextureData left, VkRender::TextureData right, VkRender::TextureData depth);

private:
    cv::Mat m_Rotation;
    cv::Mat m_Translation;
    cv::Mat m_Pose;
    cv::Mat m_Rpose;
    cv::Mat m_FramePose;
    cv::Mat m_FramePose32;
    cv::Mat m_Trajectory;
    cv::Mat m_Points4D, m_Points3D;
    cv::Mat m_ImageLeft_t0, m_ImageLeft_t1;
    uint32_t trackedFrames = 0;

    std::vector<cv::Scalar> colors{};
    FeatureSet featureSet{};

    void matchingFeatures(std::vector<cv::Point2f> *pointsLeft_t0, std::vector<cv::Point2f> *pointsLeft_t1,
                          FeatureSet *currentVOFeatures);
};


#endif //MULTISENSE_VIEWER_VISUALODOMETRY_H
