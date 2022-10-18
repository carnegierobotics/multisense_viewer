//
// Created by magnus on 10/11/22.
//

#ifndef MULTISENSE_VIEWER_VISUALODOMETRY_H
#define MULTISENSE_VIEWER_VISUALODOMETRY_H

#include "opencv4/opencv2/opencv.hpp"
#include "MultiSense/src/Core/Definitions.h"
#include "Feature.h"
#include "MultiSense/MultiSenseTypes.hh"

class VisualOdometry {

public:
    VisualOdometry();
    glm::vec3 update(VkRender::TextureData left, VkRender::TextureData right, VkRender::TextureData depth);
    void setPMat(crl::multisense::image::Calibration calibration, float tx);

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
    cv::Mat m_ImageRight_t0, m_ImageRight_t1;
    cv::Mat m_ImageDepth_t0, m_ImageDepth_t1;

    cv::Mat m_PLeft, m_PRight;
    uint32_t trackedFrames = 0;

    std::vector<cv::Scalar> colors{};
    FeatureSet featureSet{};

    int m_FrameID = 0;

    void matchingFeatures(std::vector<cv::Point2f> *pointsLeft_t0, std::vector<cv::Point2f> *pointsLeft_t1,
                          FeatureSet *currentVOFeatures);

    bool addNewFeatures = true;

    void runVO(cv::Mat imageLeft_t1, cv::Mat imageRight_t1, glm::vec3 *pos);
};


#endif //MULTISENSE_VIEWER_VISUALODOMETRY_H
