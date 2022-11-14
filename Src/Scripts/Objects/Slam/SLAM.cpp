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
    cv::Mat kernel3 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    GSlam::parseAndSortFileNames(&leftFileNames, &rightFileNames, &depthFileNames);
    cv::Mat leftImg = cv::imread(leftFileNames[frame], cv::IMREAD_GRAYSCALE);
    cv::Mat depthImg = cv::imread(depthFileNames[frame], cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    cv::Mat rightImg = cv::imread(rightFileNames[frame], cv::IMREAD_GRAYSCALE);
    filter2D(leftImg, leftImg, -1, kernel3, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    filter2D(rightImg, rightImg, -1, kernel3, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    m_FeatureLeftMap.push(GSlam::getFeatures(leftImg));
    m_FeatureRightMap.push(GSlam::getFeatures(rightImg));
    m_LImageQ.push(leftImg);
    m_RImageQ.push(rightImg);
    cv::goodFeaturesToTrack(leftImg, p0, 100, 0.5, 8, cv::Mat(), 7, false, 0.04);
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
    m_ORBDetector = cv::ORB::create();
    // Create some random colors
    cv::RNG rng;
    for (int i = 0; i < 1000; i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.emplace_back(r, g, b);
    }
    mask = cv::Mat::zeros(leftImg.size(), CV_8UC3);
}

void SLAM::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (selectedPreviewTab != TAB_3D_POINT_CLOUD)
        return;

    if (b)
        m_Model->draw(commandBuffer, i);
}

void SLAM::update() {
    if (selectedPreviewTab != TAB_3D_POINT_CLOUD)
        return;
    cv::Mat kernel3 = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

    cv::Mat leftImg;// = cv::imread(leftFileNames[frame], cv::IMREAD_GRAYSCALE);
    cv::Mat rightImg;// = cv::imread(rightFileNames[frame], cv::IMREAD_GRAYSCALE);
    cv::Mat depthImg;
    const auto& conf = renderData.crlCamera->get()->getCameraInfo(0).imgConf;
    auto tex = VkRender::TextureData(AR_POINT_CLOUD, conf.width(), conf.height(), true);
    if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Left", &tex, 0)) {
         leftImg = cv::Mat(cv::Size(conf.width(), conf.height()), CV_8UC1, tex.data);
        filter2D(leftImg, leftImg, -1, kernel3, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    }

    auto depthTex = VkRender::TextureData(AR_DISPARITY_IMAGE, conf.width(), conf.height());
    if (renderData.crlCamera->get()->getCameraStream("Disparity Left", &depthTex, 0)) {
        depthImg = cv::Mat(cv::Size(conf.width(), conf.height()), CV_16FC1, tex.data, true);

    }
    auto rightTex = VkRender::TextureData(AR_DISPARITY_IMAGE, conf.width(), conf.height(), true);
    if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Right", &depthTex, 0)) {
        rightImg = cv::Mat(cv::Size(conf.width(), conf.height()), CV_8UC1, tex.data);
        filter2D(rightImg, rightImg, -1, kernel3, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT );
    }

    if (leftImg.empty() || rightImg.empty() || depthImg.empty())
        return;

    bool anyEmpty = false;
    if (m_RImageQ.empty()){
        m_RImageQ.push(rightImg);
        anyEmpty = true;
    }
    if (m_LImageQ.empty()){
        m_LImageQ.push(leftImg);
        anyEmpty = true;
    }
    if (anyEmpty)
        return;



    shared.frame = frame;
    std::string fileName = (leftFileNames[frame].substr(leftFileNames[frame].rfind('/') + 1));
    std::string timeAndExtension = (fileName.substr(fileName.rfind('_') + 1));
    shared.time = (timeAndExtension.substr(0, timeAndExtension.rfind('.')));
    sharedData->put(&shared, shared.time.size());

    cv::Mat prevRightImg = m_RImageQ.front();
    cv::Mat prevLeftImg =  m_LImageQ.front();
    std::vector<cv::KeyPoint> prevKeyPoints, keyPoints;
    cv::Mat prevDescriptors, descriptors;

    if (p0.size() < 100) {
        std::vector<cv::Point2f> tmp{};
        cv::goodFeaturesToTrack(leftImg, tmp, 100, 0.5, 8, cv::Mat(), 7, false, 0.04);
        // Track new features, but discard identical.
        std::vector<cv::Point2f> tmp2{};
        for (const auto &item: tmp) {
            bool newPoint = true;
            for (const auto &point: p0) {
                if (point == item) {
                    newPoint = false;
                }
            }
            if (newPoint)
                tmp2.emplace_back(item);
        }
        p0.insert(p0.end(), tmp2.begin(), tmp2.end());
    }
    cv::Mat leftColor = leftImg.clone(), rightColor = rightImg.clone();
    cv::cvtColor(leftColor, leftColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rightColor, rightColor, cv::COLOR_GRAY2BGR);
    // calculate optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    cv::calcOpticalFlowPyrLK(prevLeftImg, leftImg, p0, p1, status, err, cv::Size(15, 15), 3, criteria);
    std::vector<cv::Point2f> good_new;
    for (uint i = 0; i < p0.size(); i++) {
        if (status[i] == 1) {
            good_new.push_back(p1[i]);
            //cv::line(mask, p1[i], p0[i], colors[i], 2);
            //cv::circle(leftColor, p1[i], 5, colors[i], -1);
        }
    }
    p0 = good_new;
    std::vector<cv::KeyPoint> keyPointsLeft, keyPointsRight;
    // Find points in right along the epipolar line. Discard points that were not matches
    // 1. Find descriptors in left image using a mask
    cv::Mat rightImageDescriptorMask = cv::Mat::zeros(rightImg.size(), rightImg.type());
    cv::Mat leftImageDescriptorMask = cv::Mat::zeros(leftImg.size(), leftImg.type());
    int window = 2;
    for (const auto &item: p0) {
        for (int i = -window; i <= window; ++i) {
            for (int j = -window; j <= window; ++j) {
                cv::Point2f pt = item + cv::Point2f((float) i, (float) j);
                if (pt.x > 0 && pt.y > 0 && pt.x < (float) leftImg.cols && pt.y < (float) leftImg.rows)
                    leftImageDescriptorMask.at<unsigned char>(pt.y, pt.x) = 255;
            }
        }
    }
    cv::imshow("leftImageDescriptorMask:", leftImageDescriptorMask);

    /*
    cv::Mat descriptorLeft, descriptorRight;
    m_ORBDetector->detectAndCompute(leftImg, leftImageDescriptorMask, keyPointsLeft, descriptorLeft);

    for (const auto &pt: p0) {
        for (int col = 0; col < rightImageDescriptorMask.cols; ++col) {
            for (int row = 0; row < rightImageDescriptorMask.rows; ++row) {
                if (row == (int) pt.y)
                    rightImageDescriptorMask.at<unsigned char>(row, col) = 255;
            }
        }
    }
    m_ORBDetector->detectAndCompute(rightImg, rightImageDescriptorMask, keyPointsRight, descriptorRight);
    cv::imshow("rightImageDescriptorMask:", rightImageDescriptorMask);
    // Use the matches to triangulate
    // do the 3D-to-2-D correspondence with solvePnPRansac
    // Check results
*/
    //Initialize the stereo block matching object
    auto sgbm = cv::StereoSGBM::create(0, 128, 7, 0, 0, 0, 0, 0, 0, 0, cv::StereoSGBM::MODE_SGBM);
    cv::Mat disp;
    sgbm->compute(leftImg, rightImg, disp);

    //Normalize the image for representation
    //cv::equalizeHist(disp, disp) ;
    std::vector<cv::Point2f> pointsLeft, pointsRight;

    double minVal, maxVal;
    minMaxLoc( disp, &minVal, &maxVal);
    cv::Mat disp8U;
    disp.convertTo(disp8U, CV_8UC1, 255/(maxVal - minVal));
    for (const auto &item: p0){
        auto disparity = disp8U.at<uchar>(item.y, item.x);
        // If disparity is valid
        if (item.x > 128 && disparity > 5){
            pointsLeft.emplace_back(item);
            pointsRight.emplace_back(cv::Point2f(item.x - disparity, item.y));
        }
    }

    cv::normalize(disp, disp, 0, 65535, cv::NORM_MINMAX);
    cv::imshow("Disp", disp8U);

    for (int i = 0; i < pointsLeft.size(); ++i) {
        cv::line(leftColor, pointsRight[i], pointsLeft[i], cv::Scalar(30.0f, 30.0f, 255.0f), 2);
    }

    cv::imshow("Lines", leftColor);

    /*
    cv::Mat imageKeyPointsLeft;
    cv::drawKeypoints(leftImg, keyPointsLeft, imageKeyPointsLeft, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Keypoitns:", imageKeyPointsLeft);
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch> > matches;
    matcher.knnMatch(descriptorLeft, descriptorRight, matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for (auto &knn_match: matches) {
        if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }
    cv::Mat img_matches1;
    cv::drawMatches(leftImg, keyPointsLeft, rightImg, keyPointsRight, good_matches, img_matches1);

    cv::imshow("img_matches1", img_matches1);
    std::vector<cv::Point2f> pointsLeft, pointsRight;
    for (auto &goodMatch: good_matches) {
        pointsLeft.emplace_back(keyPointsLeft[goodMatch.queryIdx].pt);
        pointsRight.emplace_back(keyPointsRight[goodMatch.trainIdx].pt);
    }
          */

    if (pointsLeft.size() >= 4) {
        cv::Mat img;
        cv::add(leftColor, mask, img);
        cv::imshow("Frame", img);
        printf("Good features: %zu\n", good_new.size());
        //cv::imshow("left", leftColor);
        //cv::imshow("right", rightColor);
        cv::Mat points3D_t0, points4D_t0;
        cv::triangulatePoints(m_PLeft, m_PRight, pointsLeft, pointsRight, points4D_t0);
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
        int flags = cv::SOLVEPNP_ITERATIVE;

        cv::Mat inliers;

        cv::solvePnPRansac(points3D_t0, pointsLeft, intrinsic_matrix, distCoeffs, m_RotVec, m_Translation,
                           useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                           inliers, flags);

        criteria = cv::TermCriteria((cv::TermCriteria::EPS) + (cv::TermCriteria::MAX_ITER), 20, 0.001);
        cv::solvePnPRefineVVS(points3D_t0, pointsLeft, intrinsic_matrix, distCoeffs, m_RotVec, m_Translation, criteria);

        //displayTracking(leftImg, p0, p1);
        float length = (float) cv::norm(m_RotVec);
        glm::vec3 rotVector(m_RotVec.at<double>(0), m_RotVec.at<double>(2), -m_RotVec.at<double>(1));
        cv::Mat data = m_Rotation.t();
        m_RotationMat = glm::rotate(m_RotationMat, length, rotVector);
        m_TranslationMat = glm::translate(m_TranslationMat, glm::vec3(m_Translation.at<double>(0), m_Translation.at<double>(1),
                                                    -m_Translation.at<double>(2)));
        std::cout << "Inliners: " << inliers.rows << std::endl;

    }

    cv::waitKey(1);

    id++;
    frame++;
    m_LImageQ.push(leftImg);
    m_RImageQ.push(rightImg);
    m_LImageQ.pop();
    m_RImageQ.pop();

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
    for (const auto &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;

        auto &preview = dev.win.at(AR_PREVIEW_POINT_CLOUD);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedMode;

        if ((currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
        }
    }
}
