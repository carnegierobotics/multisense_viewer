//
// Created by magnus on 10/11/22.
//

#include "VisualOdometry.h"
#include "thread"

VisualOdometry::VisualOdometry() {
    m_Rotation = cv::Mat::eye(3, 3, CV_64F);
    m_Translation = cv::Mat::zeros(3, 1, CV_64F);

    m_Pose = cv::Mat::zeros(3, 1, CV_64F);
    m_Rpose = cv::Mat::eye(3, 3, CV_64F);

    m_FramePose = cv::Mat::eye(4, 4, CV_64F);
    m_FramePose32 = cv::Mat::eye(4, 4, CV_32F);

    std::cout << "frame_pose " << m_FramePose << std::endl;
    m_Trajectory = cv::Mat::zeros(1000, 1200, CV_8UC3);
}

glm::vec3 VisualOdometry::update(VkRender::TextureData left, VkRender::TextureData right, VkRender::TextureData depth) {
    /*std::string filepath = "/home/magnus/PycharmProjects/simple_visual_odometry/data/data_odometry_gray/dataset/sequences/00/";
    char file[200];
    sprintf(file, "image_0/%06d.png", m_FrameID);

    // sprintf(file, "image_0/%010d.png", frame_id);
    std::string filename = filepath + std::string(file);
    cv::Mat imageLeft = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    sprintf(file, "image_1/%06d.png", m_FrameID);

    // sprintf(file, "image_0/%010d.png", frame_id);
    filename = filepath + std::string(file);
    cv::Mat imageRight = cv::imread(filename, cv::IMREAD_GRAYSCALE);
*/
    m_ImageLeft_t1 = cv::Mat(left.m_Height, left.m_Width, CV_8UC1, left.data);
    m_ImageRight_t1 = cv::Mat(right.m_Height, right.m_Width, CV_8UC1, right.data);
    runVO(m_ImageLeft_t1, m_ImageRight_t1);
}

void VisualOdometry::runVO( cv::Mat imageLeft_t1,  cv::Mat imageRight_t1){
    m_ImageLeft_t1 = imageLeft_t1;
    m_ImageRight_t1 = imageRight_t1;

    if (m_FrameID == 0) {
        m_ImageLeft_t0 = m_ImageLeft_t1.clone();
        m_ImageRight_t0 = m_ImageRight_t1.clone();
        m_ImageDepth_t0 = m_ImageDepth_t1.clone();
        m_FrameID++;
        return;
    }

    /** FIND FEATURES **/

    addNewFeatures = featureSet.points.size() < 2000;
    if (addNewFeatures) {
        std::vector<cv::Point2f> points_new;
        featureDetectionFast(m_ImageLeft_t0, points_new);
        featureSet.points.insert(featureSet.points.end(), points_new.begin(), points_new.end());
        std::vector<int> ages_new(points_new.size(), 0);
        featureSet.ages.insert(featureSet.ages.end(), ages_new.begin(), ages_new.end());
    }

    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0;
    std::vector<cv::Point2f> pointsLeft_t1, pointsRight_t1;

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    int bucket_size = m_ImageLeft_t0.rows / 10;
    int features_per_bucket = 1;
    bucketingFeatures(m_ImageLeft_t0, featureSet, bucket_size, features_per_bucket);

    pointsLeft_t0 = featureSet.points;


    cv::Mat imageLeftColor_t1;
    cv::cvtColor(m_ImageLeft_t1, imageLeftColor_t1, cv::COLOR_GRAY2BGR, 3);

    cv::Mat mask = cv::Mat::zeros(imageLeftColor_t1.size(), imageLeftColor_t1.type());

    std::vector<cv::Point2f> pointsLeftReturn_t0;   // feature points to check cicular mathcing validation
    // --------------------------------------------------------
    // Feature tracking using Optical flow PyrLK
    // --------------------------------------------------------
    std::vector<uchar> status0, status1, status2, status3;

    opticalFlow(m_ImageLeft_t0, m_ImageRight_t0, &pointsLeft_t0, &pointsRight_t0, &status0);
    opticalFlow(m_ImageRight_t0, m_ImageRight_t1, &pointsRight_t0, &pointsRight_t1, &status1);
    opticalFlow(m_ImageRight_t1, m_ImageLeft_t1, &pointsRight_t1, &pointsLeft_t1, &status2);
    opticalFlow(m_ImageLeft_t1, m_ImageLeft_t0, &pointsLeft_t1, &pointsLeftReturn_t0, &status3);

    deleteUnmatchFeaturesCircle(pointsLeft_t0, pointsRight_t0, pointsRight_t1, pointsLeft_t1, pointsLeftReturn_t0,
                                status0, status1, status2, status3, featureSet.ages);


    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 0);
    removeInvalidPoints(pointsLeft_t0, status);
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    featureSet.points = pointsLeft_t1;


    // Triangulating
    // ---------------------
    cv::Mat points3D_t0, points4D_t0;
    if (pointsLeft_t0.size() < 4 && pointsRight_t0.size() < 4){
        m_ImageLeft_t0 = m_ImageLeft_t1.clone();
        m_ImageRight_t0 = m_ImageRight_t1.clone();
        m_ImageDepth_t0 = m_ImageDepth_t1.clone();
        m_FrameID++;
        return;
    }

    cv::triangulatePoints(m_PLeft, m_PRight, pointsLeft_t0, pointsRight_t0, points4D_t0);
    cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

    trackingFrame2Frame(m_PLeft, m_PRight, pointsLeft_t0, pointsLeft_t1, points3D_t0, m_Rotation, m_Translation,
                        false);
    displayTracking(m_ImageLeft_t1, pointsLeft_t0, pointsLeft_t1);

    cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(m_Rotation);


    cv::Mat rigid_body_transformation;

    if (abs(rotation_euler[1]) < 0.1 && abs(rotation_euler[0]) < 0.1 && abs(rotation_euler[2]) < 0.1) {
        integrateOdometryStereo(m_FrameID, rigid_body_transformation, m_FramePose, m_Rotation, m_Translation);

    } else {
        std::cout << "Too large rotation" << std::endl;
    }

    //std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;
    //std::cout << "rotation: " << rotation_euler << std::endl;
    //std::cout << "translation: " << m_Translation.t() << std::endl;
    //std::cout << "frame_pose" << m_FramePose << std::endl;

    cv::Mat xyz = m_FramePose.col(3).clone();
    display(m_FrameID, m_Trajectory, xyz);


    int x = int(xyz.at<double>(0));
    int z = int(xyz.at<double>(2));
    //printf("Translation (x, z) (%d, %d)\n", x, z);


    m_ImageLeft_t0 = m_ImageLeft_t1.clone();
    m_ImageRight_t0 = m_ImageRight_t1.clone();
    m_ImageDepth_t0 = m_ImageDepth_t1.clone();
    m_FrameID++;
    cv::waitKey(1);
}

void VisualOdometry::setPMat(crl::multisense::image::Calibration calib, float tx) {

/*
    float fx = 718.85f;
    float fy = 718.85f;
    float cx = 607.19f;
    float cy = -386.144;

    tx = 386.15
 */


    float fx = calib.left.P[0][0] / 2;
    float fy = calib.left.P[1][1] / 2;
    float cx = calib.left.P[0][2] / 2;
    float cy = calib.left.P[1][2] / 2;

    m_PLeft = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);


    fx = calib.right.P[0][0] / 2;
    fy = calib.right.P[1][1] / 2;
    cx = calib.right.P[0][2] / 2;
    cy = calib.right.P[1][2] / 2;


     m_PRight = (cv::Mat_<float>(3, 4) << fx, 0., cx, fx* tx, 0., fy, cy, 0., 0, 0., 1., 0.);


}


