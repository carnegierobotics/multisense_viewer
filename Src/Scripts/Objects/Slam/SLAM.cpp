//
// Created by magnus on 10/12/22.
//

#include "SLAM.h"
#include "opencv2/opencv.hpp"
#include "MultiSense/Src/VO/LazyCSV.h"
#include <glm/gtx/string_cast.hpp>
#include <opencv2/stereo.hpp>


void SLAM::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    m_Model = std::make_unique<glTFModel::Model>(renderUtils.device);
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    m_Model->scale(glm::vec3(0.001f, 0.001f, 0.001f));
    m_Model->loadFromFile(Utils::getAssetsPath() + "Models/camera.gltf", renderUtils.device,
                          renderUtils.device->m_TransferQueue, 1.0f);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};
    // Obligatory call to prepare render resources for glTFModel.
    m_Model->createRenderPipeline(renderUtils, shaders);

    // Load t0 Images
    GSlam::parseAndSortFileNames(&leftFileNames, &rightFileNames, &depthFileNames);
    cv::Mat leftImg = cv::imread(leftFileNames[frame], cv::IMREAD_GRAYSCALE);
    cv::Mat depthImg = cv::imread(depthFileNames[frame], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    cv::Mat rightImg = cv::imread(rightFileNames[frame], cv::IMREAD_GRAYSCALE);
    m_FeatureLeftMap.push(GSlam::getFeatures(leftImg));
    m_FeatureRightMap.push(GSlam::getFeatures(rightImg));
    m_LImageQ.push(leftImg);
    m_RImageQ.push(rightImg);

    /*
    cv::drawKeypoints(leftImg, features.keypoint, leftImg);
    cv::imshow("leftImg", leftImg);
    cv::waitKey(0);
     */
    id++;
    frame++;
    float fx = 868.246;
    float fy = 868.246;
    float cx = 516.0;
    float cy = 386.0;
    m_PLeft = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
    fx = 868.246;
    fy = 868.246;
    cx = 516.0;
    cy = 386.0;
    m_PRight = (cv::Mat_<float>(3, 4) << fx, 0., cx, -78.045330571, 0., fy, cy, 0., 0, 0., 1., 0.);

    m_Rotation = cv::Mat::eye(3, 3, CV_64F);
    m_Translation = cv::Mat::zeros(3, 1, CV_64F);
    m_RotVec = cv::Mat::zeros(3, 1, CV_64F);

    m_RotationMat = glm::mat4(1.0f);
    //m_TranslationMat = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    m_TranslationMat = glm::translate(glm::mat4(1.0f), glm::vec3(-2.67f, 0.396f, 4.400f));

    m_Trajectory = cv::Mat::zeros(1000, 1200, CV_8UC3);

    sharedData->destination = "GroundTruthModel";

    orb = cv::ORB::create();

}

void SLAM::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b)
        m_Model->draw(commandBuffer, i);
}

void SLAM::update() {
    shared.frame = frame;
    std::string fileName = (leftFileNames[frame].substr(leftFileNames[frame].rfind('/') + 1));
    std::string timeAndExtension = (fileName.substr(fileName.rfind('_') + 1));
    shared.time = (timeAndExtension.substr(0, timeAndExtension.rfind('.')));

    sharedData->put(&shared, shared.time.size());

    cv::Mat leftImg = cv::imread(leftFileNames[frame], cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg = cv::imread(rightFileNames[frame], cv::IMREAD_GRAYSCALE);
    //cv::Mat depth = cv::imread(depthFileNames[frame], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);


    if (featureSet.points.size() < 4000) {
        std::vector<cv::Point2f> points_new;
        featureDetectionFast(m_LImageQ.front(), points_new);
        featureSet.points.insert(featureSet.points.end(), points_new.begin(), points_new.end());
        std::vector<int> ages_new(points_new.size(), 0);
        featureSet.ages.insert(featureSet.ages.end(), ages_new.begin(), ages_new.end());
    }

    std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0;
    std::vector<cv::Point2f> pointsLeft_t1, pointsRight_t1;

    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    int bucket_size = m_LImageQ.front().rows / 50;
    int features_per_bucket = 1;
    bucketingFeatures(m_LImageQ.front(), featureSet, bucket_size, features_per_bucket);

    /** Discard points that are far away from the sensor **/
    pointsLeft_t0 = featureSet.points;

    cv::Mat imageLeftColor_t1;
    cv::cvtColor(m_LImageQ.front(), imageLeftColor_t1, cv::COLOR_GRAY2BGR, 3);

    std::vector<cv::Point2f> pointsLeftReturn_t0;   // feature points to check cicular mathcing validation
    // --------------------------------------------------------
    // Feature tracking using Optical flow PyrLK
    // --------------------------------------------------------
    std::vector<uchar> status0, status1, status2, status3;

    opticalFlow(m_LImageQ.front(), m_RImageQ.front(), &pointsLeft_t0, &pointsRight_t0, &status0);
    opticalFlow(m_RImageQ.front(), rightImg, &pointsRight_t0, &pointsRight_t1, &status1);
    opticalFlow(rightImg, leftImg, &pointsRight_t1, &pointsLeft_t1, &status2);
    opticalFlow(leftImg, m_LImageQ.front(), &pointsLeft_t1, &pointsLeftReturn_t0, &status3);

    deleteUnmatchFeaturesCircle(pointsLeft_t0, pointsRight_t0, pointsRight_t1, pointsLeft_t1, pointsLeftReturn_t0,
                                status0, status1, status2, status3, featureSet.ages);


    std::vector<bool> status;
    checkValidMatch(pointsLeft_t0, pointsLeftReturn_t0, status, 0);
    removeInvalidPoints(pointsLeft_t0, status);
    removeInvalidPoints(pointsLeft_t1, status);
    removeInvalidPoints(pointsRight_t0, status);
    removeInvalidPoints(pointsRight_t1, status);

    featureSet.points = pointsLeft_t1;

    if (featureSet.points.size() < 50) {
        m_LImageQ.push(leftImg);
        m_RImageQ.push(rightImg);

        m_LImageQ.pop();
        m_RImageQ.pop();
        return;
    }
    // Triangulating
    // ---------------------

    cv::Mat points3D_t0, points4D_t0;
    cv::triangulatePoints(m_PLeft, m_PRight, pointsLeft_t0, pointsRight_t0, points4D_t0);
    cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << -0.197412f, 0.236726f, 0.0, 0.0, 0.0);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);

    cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) <<
                                                      m_PLeft.at<float>(0, 0), m_PLeft.at<float>(0,
                                                                                                 1), m_PLeft.at<float>(
            0, 2),
            m_PLeft.at<float>(1, 0), m_PLeft.at<float>(1, 1), m_PLeft.at<float>(1, 2),
            m_PLeft.at<float>(2, 0), m_PLeft.at<float>(2, 1), m_PLeft.at<float>(2, 2));

    int iterationsCount = 2000;        // number of Ransac iterations.
    float reprojectionError = .1;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.999;          // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags = cv::SOLVEPNP_P3P;


    cv::Mat inliers;
    cv::solvePnPRansac(points3D_t0, pointsLeft_t1, intrinsic_matrix, distCoeffs, m_RotVec, m_Translation,
                       useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                       inliers, flags);

    displayTracking(leftImg, pointsLeft_t0, pointsLeft_t1);

    float length = (float) cv::norm(m_RotVec);
    glm::vec3 rotVector(m_RotVec.at<double>(0), -m_RotVec.at<double>(1), -m_RotVec.at<double>(2));

    cv::Mat data = m_Rotation.t();

    m_RotationMat = glm::rotate(m_RotationMat, length, rotVector);
    m_TranslationMat = glm::translate(m_TranslationMat,
                                      glm::vec3(m_Translation.at<double>(0), -m_Translation.at<double>(1),
                                                -m_Translation.at<double>(2)));
    cv::waitKey(1);

    id++;
    frame++;
    m_LImageQ.push(leftImg);
    m_RImageQ.push(rightImg);
    m_LImageQ.pop();
    m_RImageQ.pop();


    std::cout << "Inliners: " << inliers.rows << std::endl;
    std::cout << glm::to_string(m_TranslationMat) << std::endl;
    std::cout << glm::to_string(m_RotationMat) << std::endl << std::endl;


    VkRender::UBOMatrix mat{};
    mat.model = m_TranslationMat * m_RotationMat;
    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;

}


void SLAM::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

    }
}

void SLAM::fromCV2GLM(const cv::Mat &cvmat, glm::mat4 *glmmat) {
    if (cvmat.cols != 4 || cvmat.rows != 4 || cvmat.type() != CV_32FC1) {
        std::cout << "Matrix conversion error!" << std::endl;
        return;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat.data, 16 * sizeof(float));
}