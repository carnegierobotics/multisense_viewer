//
// Created by magnus on 10/12/22.
//
#include <TinyTIFF/src/tinytiffwriter.h>
#include <random>

#include "Viewer/Scripts/Objects/Video/RecordExposure.h"

void RecordExposure::setup() {
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    threadPool = std::make_unique<VkRender::ThreadPool>();



}

void RecordExposure::update() {
    if (!saveImage)
        return;

    // For each enabled source in all windows
    std::string src = "Luma Left";
    // For each remote head index

    const auto &conf = renderData.crlCamera->get()->getCameraInfo(0).imgConf;
    auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(), conf.height(), true);

    std::mt19937 rng = std::mt19937(renderData.scriptDrawCount);
    std::uniform_int_distribution<int> gen = std::uniform_int_distribution<int>(95, 105);

    // Set exposure
    ExposureParams params{};
    params.autoExposure = false;
    //params.exposure =  static_cast<uint32_t>(exposure);

    //if (renderData.crlCamera->get()->setExposureParams(params, 0)) {
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //std::cout << "Set exposure: " << params.exposure << std::endl;

        if (renderData.crlCamera->get()->getCameraStream(src, tex.get(), 0)) {
            saveImageToFile(Utils::CRLSourceToTextureType(src), saveImagePath, src + " exposures",
                            0, tex, false, params.exposure);

            // New exposure for next frame

            int val = gen(rng);
            //exposure = params.exposure * ((float)val / 100.0f); // uniform, unbiased
        } else {
            std::cerr << "Failed to get image.." << std::endl;
        }
//    } else {
//        std::cerr << "Failed to set exposure: " << exposure << std::endl;
//    }
}

void RecordExposure::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        saveImagePath = dev.outputSaveFolder;
        saveImage = dev.isRecording;
        isRemoteHead = dev.isRemoteHead;
    }
}


void RecordExposure::saveImageToFile(CRLCameraDataType type, const std::string &path, const std::string &stringSrc,
                                     short remoteHead,
                                     std::shared_ptr<VkRender::TextureData> &ptr, bool isRemoteHead, uint32_t exposure) {

    std::string separator = "/";

    std::string directory, remoteHeadDir;
    directory = std::string(path);
    std::string fileName = std::to_string(ptr->m_Id);
    if (isRemoteHead) {
        remoteHeadDir = path + separator + std::to_string(remoteHead);
        directory = remoteHeadDir + separator + std::string(stringSrc);
    } else
        directory = std::string(path) + separator + stringSrc;

    std::string filePath = directory + std::string(separator);

    // Create Source name directory
    int check = mkdir(directory.c_str(), 0777);
    if (check == 0)
        Log::Logger::getInstance()->info("Created directory {}", directory);


    std::string fullPathName = filePath + fileName + ".png";
    // Check if file exists. Otherwise do nothing
    std::ifstream fin(fullPathName.c_str());
    if (fin.good()) {
        fin.close();
        return;
    }
    // Create file


    switch (type) {
        case AR_POINT_CLOUD:
            break;
        case AR_GRAYSCALE_IMAGE: {
            std::ofstream output(fullPathName);
            output.close();
            Log::Logger::getInstance()->info("Saving Frame: {} from source: {}", ptr->m_Id, stringSrc);

            stbi_write_png((fullPathName).c_str(), ptr->m_Width, ptr->m_Height, 1, ptr->data,
                           ptr->m_Width);

            std::ofstream exposureTimes((filePath + "times.csv").c_str(), std::ios::app);
            exposureTimes << fileName << "," << std::to_string(exposure) << ",\n";
            exposureTimes.close();
        }
            break;
        case AR_DISPARITY_IMAGE: {
        }
            break;
        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_CAMERA_IMAGE_NONE:
            break;
        case AR_COLOR_IMAGE:
            break;
    }
}
