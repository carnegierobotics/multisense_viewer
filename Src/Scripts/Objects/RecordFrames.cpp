//
// Created by magnus on 10/12/22.
//

#include "RecordFrames.h"


void RecordFrames::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    threadPool = std::make_unique<ThreadPool>();

}

void RecordFrames::update() {
    if (!saveImage)
        return;
    // There might be some delay for when the camera actually sets the resolution therefore add this check so we dont render to a texture that does not match the actual camere frame size

    for (const auto &src: sources) {
        const auto &conf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
        auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(), conf.height(), false);

        if (renderData.crlCamera->get()->getCameraStream(src, tex.get(), remoteHeadIndex)) {
            if (threadPool->getTaskListSize() < STACK_SIZE_100) {
                Log::Logger::getInstance()->info("Saving image {} from source {}", tex->m_Id, src);
                threadPool->Push(Utils::saveImageToFile2, Utils::CRLSourceToTextureType(src), saveImagePath, src, tex);
            }
        }

    }
}

void RecordFrames::onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
    for (const MultiSense::Device &dev: *uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        sources.clear();
        for (auto window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource))
                sources.emplace_back(window.second.selectedSource);
        }


        if (!initializeBuffers) {
            for (auto availSources: dev.win.at(0).availableSources) {
                // Create buffers large enough to contain the largest image
                dataBufferMap[availSources] = (uint8_t *) malloc(1920 * 1200 * 2);

                if (availSources == "Color Rectified Aux" || availSources == "Color Rectified"){
                    dataBufferMapSecondary[availSources] = (uint8_t *) malloc(1920 * 1200 * 2);
                }

                initializeBuffers = true;
            }
        }

        saveImage = dev.isRecording;
        saveImagePath = dev.outputSaveFolder;
    }
}

