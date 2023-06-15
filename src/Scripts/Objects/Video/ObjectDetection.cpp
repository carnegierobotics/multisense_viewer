/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Objects/Example.h
 *
 * Copyright 2023
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2023-06-13, mgjerde@carnegierobotics.com, Created file.
 **/

#include <cstdlib>
#include <cstring>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/ocl.hpp> // check OCL


#include "Viewer/Core/Definitions.h"
#include "Viewer/Scripts/Objects/Video/ObjectDetection.h"

void ObjectDetection::setup() {


    std::string modelPath = Utils::getAssetsPath() / "ML/yolov7.onnx";
    int inputHeight = 640;
    int inputWidth = 640;

    YoloV7Config conf = {0.3, 0.5, modelPath, inputHeight, inputWidth};
    yolo = std::make_unique<YOLOV7>(conf);
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_FREERATIO);

    cv::dnn::Net net;
    // Check if OpenCL is available
    if (cv::ocl::haveOpenCL())
    {
        std::cout << "OpenCL is available" << std::endl;
        // Set backend and target to OpenCL
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
    else
    {
        std::cout << "OpenCL is not available" << std::endl;
    }
    auto var = cv::dnn::getAvailableBackends();

    auto target = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_VKCOM);

    target = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);

    int a = 0;

}

void ObjectDetection::update() {

    const auto &conf = renderData.crlCamera->getCameraInfo(0).imgConf;
    uint32_t width = conf.width(), height = conf.height();
    auto tex = std::make_shared<VkRender::TextureData>(CRL_GRAYSCALE_IMAGE, conf.width(),
                                                       conf.height(), false, true);


    if (key && renderData.crlCamera->getCameraStream("Luma Rectified Left", tex.get(), 0)) {

        cv::Mat gray(height, width, CV_8UC1, tex->data);
        std::string imgpath = Utils::getAssetsPath() / "ML/image.jpg";

        cv::Mat img = cv::imread(imgpath);
        //cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
        yolo->detect(img);

        static const std::string kWinName = "Deep learning object detection in OpenCV";

        cv::imshow(kWinName, img);
        cv::resizeWindow(kWinName, 960, 600);
        cv::waitKey(1);
    }
    else
        cv::destroyAllWindows();

}

void ObjectDetection::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    if (uiHandle->input->getButtonDown(GLFW_KEY_D))
        key = !key;


}

void ObjectDetection::onDestroy() {
    cv::destroyAllWindows();
}
