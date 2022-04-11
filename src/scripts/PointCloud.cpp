//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"


void PointCloud::setup() {

    model = new CRLCameraModels::Model(renderUtils.device);
    this->drawModel = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");


    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);
}


void PointCloud::update() {

    if (camera == nullptr)
        return;

    stream = &camera->getImage()[crl::multisense::Source_Disparity_Left];

    if (stream->source == crl::multisense::Source_Disparity_Left) {

        cv::Mat disparityMat(stream->height, stream->width, CV_16UC1,
                             const_cast<void *>(stream->imageDataP));

        disparityMat.convertTo(disparityFloatMat, CV_32FC1, 1.0 / 16.0);

        crl::multisense::image::Config c = camera->getImageConfig();


        m_qMatrix = cv::Mat(4, 4, CV_32F, 0.0);

        m_qMatrix.at<float>(0, 0) = c.fy() * c.tx();
        m_qMatrix.at<float>(1, 1) = c.fx() * c.tx();
        m_qMatrix.at<float>(0, 3) = -c.fy() * c.cx() * c.tx();
        m_qMatrix.at<float>(1, 3) = -c.fx() * c.cy() * c.tx();
        m_qMatrix.at<float>(2, 3) = c.fx() * c.fy() * c.tx();
        m_qMatrix.at<float>(3, 2) = -c.fy();
        m_qMatrix.at<float>(3, 3) = c.fy() * (0.0f);


        reprojectImageTo3D(disparityFloatMat, cloudMat, m_qMatrix, true);

        cv::Vec3f CloudPt;
        float X, Y, Z;
        CRLBaseCamera::PointCloudData *meshData = camera->getStream();

        auto *p = (CRLCameraModels::Model::Vertex *) meshData->vertices;


        for (int r = 0; r < cloudMat.rows; r++) {
            for (int c = 0; c < cloudMat.cols; c++) {
                CloudPt = cloudMat.at<cv::Vec3f>(r, c);
                X = (float) (CloudPt[0]);           // Left-Right in camera frame
                Y = (float) (-CloudPt[1]);          // Up-down in camera frame
                Z = (float) (-CloudPt[2]);          // More negative is farther from the camera

                point = (cloudMat.cols * r) + c;
                p[point].pos = glm::vec3(0, 0, 0);


                if (X < 20 && X > -50) {
                    if (Y < 20 && Y > -50) {
                        if (Z < 20 && Z > -50) {
                            //printf("Coord: %f, %f, %f\n", X, Y, Z);
                            // pixel at (r, c) should correspond to p[index]
                            // 2, 5
                            p[point].pos = glm::vec3(X, Y, Z);
                        }
                    }
                }
            }
        }

        model->createMesh((CRLCameraModels::Model::Vertex *) meshData->vertices, meshData->vertexCount);


        // Transform pointcloud
        UBOMatrix mat{};
        mat.model = glm::mat4(1.0f);
        mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        mat.model = glm::translate(mat.model, glm::vec3(3.0f, 0.0f, 0.0f));
        auto *d = (UBOMatrix *) bufferOneData;
        d->model = mat.model;
        d->projection = renderData.camera->matrices.perspective;
        d->view = renderData.camera->matrices.view;
        drawModel = true;
    }
    else{
        drawModel = false;
    }
}


void PointCloud::onUIUpdate(UISettings *uiSettings) {
    camera = (CRLPhysicalCamera *) uiSettings->physicalCamera;

}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (this->drawModel)
        CRLCameraModels::draw(commandBuffer, i, this->model);
}