//
// Created by magnus on 10/11/22.
//

#include "VisualOdometry.h"

/*
VisualOdometry::VisualOdometry() {
    float fx = 320;
    float fy = 320;
    float cx = 320;
    float cy = 320;
    float bf = 27;

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0, 0., 1., 0.);

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    m_Rotation = cv::Mat::eye(3, 3, CV_64F);
    m_Translation = cv::Mat::zeros(3, 1, CV_64F);

    m_Pose = cv::Mat::zeros(3, 1, CV_64F);
    m_Rpose = cv::Mat::eye(3, 3, CV_64F);

    m_FramePose = cv::Mat::eye(4, 4, CV_64F);
    m_FramePose32 = cv::Mat::eye(4, 4, CV_32F);
    m_Trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);


    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageRight_t0, imageLeft_t0;

}
*/

void VisualOdometry::update(VkRender::TextureData left, VkRender::TextureData right, VkRender::TextureData depth) {
    m_ImageLeft_t1 = cv::Mat(left.m_height, left.m_width, CV_8UC1, left.data);


        //ppendNewFeatures(m_ImageLeft_t1, &featureSet);


    if (trackedFrames == 0){
        m_ImageLeft_t0 = m_ImageLeft_t1;
        trackedFrames++;
        cv::goodFeaturesToTrack(m_ImageLeft_t0, featureSet.points, 2000, 0.3, 7, cv::Mat(), 7, false, 0.04);
        return;
    }

    cv::Mat imageLeftColor_t1;
    cv::cvtColor(m_ImageLeft_t1, imageLeftColor_t1, cv::COLOR_GRAY2BGR, 3);

    std::vector<cv::Point2f> pointsLeft_t0{};
    cv::Mat mask = cv::Mat::zeros(imageLeftColor_t1.size(), imageLeftColor_t1.type());

    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<cv::Point2f>  pointsLeftReturn_t0;   // feature points to check cicular mathcing validation
    // --------------------------------------------------------
    // Feature tracking using Optical flow PyrLK
    // --------------------------------------------------------
    std::vector<uchar> status;
    std::vector<cv::Point2f> pointsLeft_t1;
    opticalFlow(m_ImageLeft_t0, m_ImageLeft_t1, &featureSet.points, &pointsLeft_t1, &status);

    std::vector<cv::Point2f> good_new;
    for(uint i = 0; i < featureSet.points.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(pointsLeft_t1[i]);
            // draw the tracks
            line(mask,pointsLeft_t1[i], featureSet.points[i], colors[i], 2);
            circle(imageLeftColor_t1, pointsLeft_t1[i], 5, colors[i], -1);
        }
    }
    cv::Mat img;
    cv::add(imageLeftColor_t1, mask, img);

    imshow("Frame", img);

    featureSet.points = good_new;
    m_ImageLeft_t0 = m_ImageLeft_t1.clone();
    //cv::waitKey(1);
}

