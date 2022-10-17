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
        auto tex = VkRender::TextureData(Utils::CRLSourceToTextureType(src), conf.width(), conf.height());

        tex.data = dataBufferMap[src];
        if (dataBufferMapSecondary.find(src) != dataBufferMapSecondary.end())
            tex.data2 = dataBufferMapSecondary[src];

        if (renderData.crlCamera->get()->getCameraStream(src, &tex, remoteHeadIndex)) {
            if (threadPool->getTaskListSize() < 100 && ids[src] != tex.m_Id) {

                Log::Logger::getInstance()->info("Saving image {} from source {}", tex.m_Id, src);

                auto* data = (uint8_t*) malloc(tex.m_Len);
                std::memcpy(data, tex.data, tex.m_Len);
                threadPool->Push(Utils::saveImageToFile, Utils::CRLSourceToTextureType(src), saveImagePath, src, tex, data,
                                 tex.data2);
                ids[src] = tex.m_Id;
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

