//
// Created by magnus on 10/12/22.
//

#include "SLAM.h"
#include "opencv2/opencv.hpp"
#include "MultiSense/Src/VO/LazyCSV.h"

void SLAM::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    m_Model = std::make_unique<glTFModel::Model>(renderUtils.device);
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    m_Model->loadFromFile(Utils::getAssetsPath() + "Models/camera.gltf", renderUtils.device,
                         renderUtils.device->m_TransferQueue, 1.0f);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("myScene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("myScene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};
    // Obligatory call to prepare render resources for glTFModel.
    m_Model->createRenderPipeline(renderUtils, shaders);

    // Load t0 Images
    GSlam::parseAndSortFileNames(&leftFileNames, &rightFileNames, &depthFileNames);
    cv::Mat leftImg = cv::imread(leftFileNames[0]);
    cv::Mat depthImg = cv::imread(depthFileNames[0]);
    cv::Mat rightImg = cv::imread(rightFileNames[0]);
    m_FeatureLeftMap[id] = GSlam::getFeatures(leftImg);
    m_FeatureRightMap[id] = GSlam::getFeatures(rightImg);
    m_LMap[id] = leftImg;
    m_DMap[id] = depthImg;
    m_RMap[id] = rightImg;
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
    m_Pose = cv::Mat::eye(4, 4, CV_64F);
    m_Trajectory = cv::Mat::zeros(1000, 1200, CV_8UC3);
    lazycsv::parser parser{ "../Slam/G0/G-0_ground_truth/gt_6DoF_gnss_and_imu.csv" };


    sharedData->destination = "Map";
}

void SLAM::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    m_Model->draw(commandBuffer, i);
}

void SLAM::update() {
    sharedData->put(&frame);
    if (id > 10) {
        id = id % 10;
    }
    /*
    printf("Reading frame %lu\n", frame);
    cv::Mat leftImg = cv::imread(leftFileNames[frame]);
    m_LMap[id] = leftImg;
    cv::Mat depthImg = cv::imread(depthFileNames[frame], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    m_DMap[id] = depthImg;
    cv::Mat rightImg = cv::imread(rightFileNames[frame]);
    m_RMap[id] = rightImg;

    m_FeatureLeftMap[id] = GSlam::getFeatures(leftImg);
    m_FeatureRightMap[id] = GSlam::getFeatures(rightImg);

    // Match against previous frame
    GSlam::FeatureMatches matchesLeft = GSlam::featureMatcher(m_FeatureLeftMap, id);
    GSlam::FeatureMatches matchesRight = GSlam::featureMatcher(m_FeatureRightMap, id);
    std::map<size_t, GSlam::FeatureSet> map;
    map[0] = m_FeatureLeftMap[id];
    map[1] = m_FeatureRightMap[id];
    GSlam::FeatureMatches matchesLR = GSlam::featureMatcher(map, 1);
    cv::Mat img_matches1;
    cv::drawMatches(leftImg, map[0].keypoint,
                    rightImg, map[1].keypoint,
                    matchesLR.match1, img_matches1);

    cv::imshow("matches", img_matches1);

    std::vector<cv::Point2f> pointsLeft, pointsRight;
    for (const auto &match: matchesLR.match1) {
        pointsLeft.emplace_back(map[0].keypoint[match.queryIdx].pt);
        pointsRight.emplace_back(map[1].keypoint[match.trainIdx].pt);
    }


    cv::Mat points3D_t0, points4D_t0;
    cv::triangulatePoints(m_PLeft, m_PRight, pointsLeft, pointsRight, points4D_t0);
    cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) <<
                                                      m_PLeft.at<float>(0, 0), m_PLeft.at<float>(0,
                                                                                                 1), m_PLeft.at<float>(
            0, 2),
            m_PLeft.at<float>(1, 0), m_PLeft.at<float>(1, 1), m_PLeft.at<float>(1, 2),
            m_PLeft.at<float>(2, 0), m_PLeft.at<float>(2, 1), m_PLeft.at<float>(2, 2));

    int iterationsCount = 1000;        // number of Ransac iterations.
    float reprojectionError = .5;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.999;          // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags = cv::SOLVEPNP_AP3P;


    cv::Mat inliers;
    cv::solvePnPRansac(points3D_t0, pointsLeft, intrinsic_matrix, distCoeffs, rvec, m_Translation,
                       useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                       inliers, flags);


    cv::Mat rigidBodyTransformation;
    cv::Mat addup = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
    cv::hconcat(m_Rotation, m_Translation, rigidBodyTransformation);
    cv::vconcat(rigidBodyTransformation, addup, rigidBodyTransformation);

    double scale = sqrt((m_Translation.at<double>(0)) * (m_Translation.at<double>(0))
                        + (m_Translation.at<double>(1)) * (m_Translation.at<double>(1))
                        + (m_Translation.at<double>(2)) * (m_Translation.at<double>(2)));

    rigidBodyTransformation = rigidBodyTransformation.inv();
    if (scale < 1){
        m_Pose = m_Pose * rigidBodyTransformation;
    } else {
        printf("Warning scale too high to be reasonable %f\n", scale);
    }
    printf("Scale: %f\n", scale);


    cv::Mat xyz = m_Pose.col(3).clone() * 50;
    glm::vec3 translation(float(xyz.at<double>(0)), float(xyz.at<double>(1)), float(xyz.at<double>(2)));

    int x = int(xyz.at<double>(0)) * 100 + 600;
    int z = int(xyz.at<double>(2)) * 100 + 300;

    std::cout << "xyz: " << xyz << std::endl;
    circle(m_Trajectory, cv::Point(x, z), 1, CV_RGB(255, 0, 0), 2);
    // rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    // sprintf(text, "FPS: %02f", fps);
    // putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
    cv::imshow("Trajectory", m_Trajectory);
    cv::waitKey(1);
    */
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, -3.0f));
    //mat.model = glm::translate(mat.model, (translation * glm::vec3(1.0f, 1.0f, 1.0f)));
    mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(0.001f, 0.001f, 0.001f));


    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;

    id++;
    frame++;
}


void SLAM::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

    }
}
