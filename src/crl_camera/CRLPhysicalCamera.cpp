//
// Created by magnus on 3/1/22.
//

#include <thread>
#include <opencv2/core/mat.hpp>
#include <bitset>
#include "CRLPhysicalCamera.h"


void CRLPhysicalCamera::connect() {
    online = CRLBaseCamera::connect(DEFAULT_CAMERA_IP);

}

void CRLPhysicalCamera::start(std::string string, std::string dataSourceStr) {

    crl::multisense::DataSource source = stringToDataSource(dataSourceStr);
    uint32_t colorSource = crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Chroma_Rectified_Aux |
                           crl::multisense::Source_Chroma_Aux |
                           crl::multisense::Source_Chroma_Left | crl::multisense::Source_Chroma_Right;
    if (source & colorSource)
        enabledSources.push_back(crl::multisense::Source_Luma_Rectified_Aux);

    enabledSources.push_back(source);

    // Set mode first
    std::string delimiter = "x";

    size_t pos = 0;
    std::string token;
    std::vector<uint32_t> widthHeightDepth;
    while ((pos = string.find(delimiter)) != std::string::npos) {
        token = string.substr(0, pos);
        widthHeightDepth.push_back(std::stoi(token));
        string.erase(0, pos + delimiter.length());
    }

    this->selectDisparities(widthHeightDepth[2]);
    this->selectResolution(widthHeightDepth[0], widthHeightDepth[1]);
    this->selectFramerate(60);

    // Start stream
    for (auto src: enabledSources) {
        bool status = cameraInterface->startStreams(src);
        printf("Started stream %s status: %d\n", dataSourceToString(src).c_str(), status);
    }


    std::thread thread_obj(CRLPhysicalCamera::setDelayedPropertyThreadFunc, this);

    thread_obj.join();
}

void CRLPhysicalCamera::setDelayedPropertyThreadFunc(void *context) {
    auto *app = static_cast<CRLPhysicalCamera *>(context);
    std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(1));
    app->modeChange = true;
    app->play = true;

}

void CRLPhysicalCamera::stop(std::string dataSourceStr) {
    // Start stream
    crl::multisense::DataSource src = stringToDataSource(dataSourceStr);

    std::vector<uint32_t>::iterator it;
    // Search and stop additional sources
    it = std::find(enabledSources.begin(), enabledSources.end(), crl::multisense::Source_Chroma_Rectified_Aux);
    if (it != enabledSources.end()){
        src |= crl::multisense::Source_Luma_Rectified_Aux;
    }
    enabledSources.clear();

    bool status = cameraInterface->stopStreams(src);
    printf("Stopped stream %s status: %d\n", dataSourceStr.c_str(), status);
    modeChange = true;
}


CRLBaseCamera::PointCloudData *CRLPhysicalCamera::getStream() {
    return meshData;
}

std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> CRLPhysicalCamera::getImage() {
    return imagePointers;
}

CRLPhysicalCamera::~CRLPhysicalCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}

CRLBaseCamera::CameraInfo CRLPhysicalCamera::getInfo() {
    return CRLBaseCamera::cameraInfo;
}

// Pick an image size
crl::multisense::image::Config CRLPhysicalCamera::getImageConfig() const {
    // Configure the sensor.
    crl::multisense::image::Config cfg;
    bool status = cameraInterface->getImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to query image config: %d\n", status);
    }

    return cfg;
}

std::unordered_set<crl::multisense::DataSource> CRLPhysicalCamera::supportedSources() {
    // this method effectively restrics the supported sources for the classice libmultisense api
    std::unordered_set<crl::multisense::DataSource> ret;
    if (cameraInfo.supportedSources & crl::multisense::Source_Raw_Left) ret.insert(crl::multisense::Source_Raw_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Raw_Right) ret.insert(crl::multisense::Source_Raw_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Left) ret.insert(crl::multisense::Source_Luma_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Right)
        ret.insert(crl::multisense::Source_Luma_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Rectified_Left)
        ret.insert(crl::multisense::Source_Luma_Rectified_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Rectified_Right)
        ret.insert(crl::multisense::Source_Luma_Rectified_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Aux)
        ret.insert(crl::multisense::Source_Chroma_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Left)
        ret.insert(crl::multisense::Source_Chroma_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Right)
        ret.insert(crl::multisense::Source_Chroma_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Left)
        ret.insert(crl::multisense::Source_Disparity_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Right)
        ret.insert(crl::multisense::Source_Disparity_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Cost)
        ret.insert(crl::multisense::Source_Disparity_Cost);
    if (cameraInfo.supportedSources & crl::multisense::Source_Raw_Aux) ret.insert(crl::multisense::Source_Raw_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Aux) ret.insert(crl::multisense::Source_Luma_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Rectified_Aux)
        ret.insert(crl::multisense::Source_Luma_Rectified_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Rectified_Aux)
        ret.insert(crl::multisense::Source_Chroma_Rectified_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Aux)
        ret.insert(crl::multisense::Source_Disparity_Aux);
    return ret;
}

std::string CRLPhysicalCamera::dataSourceToString(crl::multisense::DataSource d) {
    switch (d) {
        case crl::multisense::Source_Raw_Left:
            return "Raw Left";
        case crl::multisense::Source_Raw_Right:
            return "Raw Right";
        case crl::multisense::Source_Luma_Left:
            return "Luma Left";
        case crl::multisense::Source_Luma_Right:
            return "Luma Right";
        case crl::multisense::Source_Luma_Rectified_Left:
            return "Luma Rectified Left";
        case crl::multisense::Source_Luma_Rectified_Right:
            return "Luma Rectified Right";
        case crl::multisense::Source_Chroma_Left:
            return "Color Left";
        case crl::multisense::Source_Chroma_Right:
            return "Source Color Right";
        case crl::multisense::Source_Disparity_Left:
            return "Disparity Left";
        case crl::multisense::Source_Disparity_Cost:
            return "Disparity Cost";
        case crl::multisense::Source_Jpeg_Left:
            return "Jpeg Left";
        case crl::multisense::Source_Rgb_Left:
            return "Source Rgb Left";
        case crl::multisense::Source_Lidar_Scan:
            return "Source Lidar Scan";
        case crl::multisense::Source_Raw_Aux:
            return "Raw Aux";
        case crl::multisense::Source_Luma_Aux:
            return "Luma Aux";
        case crl::multisense::Source_Luma_Rectified_Aux:
            return "Luma Rectified Aux";
        case crl::multisense::Source_Chroma_Aux:
            return "Color Aux";
        case crl::multisense::Source_Chroma_Rectified_Aux:
            return "Color Rectified Aux";
        case crl::multisense::Source_Disparity_Aux:
            return "Disparity Aux";
        default:
            return "Unknown";
    }
}

crl::multisense::DataSource CRLPhysicalCamera::stringToDataSource(const std::string &d) {
    if (d == "Raw Left") return crl::multisense::Source_Raw_Left;
    if (d == "Raw Right") return crl::multisense::Source_Raw_Right;
    if (d == "Luma Left") return crl::multisense::Source_Luma_Left;
    if (d == "Luma Right") return crl::multisense::Source_Luma_Right;
    if (d == "Luma Rectified Left") return crl::multisense::Source_Luma_Rectified_Left;
    if (d == "Luma Rectified Right") return crl::multisense::Source_Luma_Rectified_Right;
    if (d == "Color Left") return crl::multisense::Source_Chroma_Left;
    if (d == "Source Color Right") return crl::multisense::Source_Chroma_Right;
    if (d == "Disparity Left") return crl::multisense::Source_Disparity_Left;
    if (d == "Disparity Cost") return crl::multisense::Source_Disparity_Cost;
    if (d == "Jpeg Left") return crl::multisense::Source_Jpeg_Left;
    if (d == "Source Rgb Left") return crl::multisense::Source_Rgb_Left;
    if (d == "Source Lidar Scan") return crl::multisense::Source_Lidar_Scan;
    if (d == "Raw Aux") return crl::multisense::Source_Raw_Aux;
    if (d == "Luma Aux") return crl::multisense::Source_Luma_Aux;
    if (d == "Luma Rectified Aux") return crl::multisense::Source_Luma_Rectified_Aux;
    if (d == "Color Aux") return crl::multisense::Source_Chroma_Aux;
    if (d == "Color Rectified Aux") return crl::multisense::Source_Chroma_Rectified_Aux;
    if (d == "Disparity Aux") return crl::multisense::Source_Disparity_Aux;
    throw std::runtime_error(std::string{} + "Unknown Datasource: " + d);
}

void CRLPhysicalCamera::update() {

    for (auto src: enabledSources) {
        if (src == crl::multisense::Source_Disparity_Left) {
            // Reproject camera to 3D
            stream = &imagePointers[crl::multisense::Source_Disparity_Left];

        }
    }


    crl::multisense::DataSource config;
    bool status = cameraInterface->getEnabledStreams(config) != 0;
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to query image config: %d\n", status);
    }
    std::bitset<32> y(config);
    //std::cout << y << '\n';



}

void CRLPhysicalCamera::setup() {

    meshData = new PointCloudData(960, 600);

    crl::multisense::image::Config c = cameraInfo.imgConf;


    kInverseMatrix =
              glm::mat4(
                      glm::vec4(1/c.fx(), 0, -(c.cx()*c.fx())/(c.fx() * c.fy()), 0),
                      glm::vec4(0, 1/c.fy(), -c.cy() / c.fy(), 0),
                      glm::vec4(0, 0,  1, 0),
                      glm::vec4(0, 0, 0, 1));
    /*
   kInverseMatrix = glm::mat4(glm::vec4(c.fy() * c.tx(), 0, 0, -c.fy() * c.cx() * c.tx()),
                  glm::vec4(0, c.fx() * c.tx(), 0, -c.fx() * c.cy() * c.tx()),
                  glm::vec4(0, 0, 0, c.fx() * c.fy() * c.tx()),
                  glm::vec4(0, 0, -c.fx(), c.fy() * 1));

                      kInverseMatrix =
              glm::mat4(
                      glm::vec4(1/c.fx(), 0, -(c.cx()*c.fx())/(c.fx() * c.fy()), 0),
                      glm::vec4(0, 1/c.fy(), -c.cy() / c.fy(), 0),
                      glm::vec4(0, 0,  1, 0),
                      glm::vec4(0, 0, 0, 1));
  */
    // Load calibration data


}

cv::Mat *CRLPhysicalCamera::getCloudMat() {
    return &cloudMat;
}
