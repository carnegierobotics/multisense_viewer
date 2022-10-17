//
// Created by magnus on 10/12/22.
//

#include "RecordFrames.h"


void RecordFrames::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    threadPool = std::make_unique<ThreadPool>();
}

void RecordFrames::update() {
    if (!saveImage)
        return;
    // There might be some delay for when the camera actually sets the resolution therefore add this check so we dont render to a texture that does not match the actual camere frame size
    if (renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf.width() != width) {
        emptyTexture();
        return;
    }

    for (const auto &src: sources) {
        const auto &conf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
        auto tex = VkRender::TextureData(Utils::CRLSourceToTextureType(src), conf.width(), conf.height());
        model->getTextureDataPointer(&tex);

        if (renderData.crlCamera->get()->getCameraStream(src, &tex, remoteHeadIndex)) {
            if (threadPool->getTaskListSize() < 100 && ids[src] != tex.m_Id) {

                Log::Logger::getInstance()->info("Saving image {} from source {}", tex.m_Id, src);
                void* copyData = malloc(tex.m_Len);
                std::memcpy(copyData, tex.data, tex.m_Len);

                threadPool->Push(Utils::saveImageToFile, Utils::CRLSourceToTextureType(src), saveImagePath, src, tex, copyData);
                ids[src] = tex.m_Id;
            }
        }

    }
}


void RecordFrames::emptyTexture() {
    model->modelType = textureType;
    auto imgConf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
    width = imgConf.width();
    height = imgConf.height();
    model->createEmtpyTexture(width, height, textureType);
}

void RecordFrames::onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
    for (const MultiSense::Device &dev: *uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        for (auto window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource))
                sources.emplace_back(window.second.selectedSource);
        }

        saveImage = dev.isRecording;
        saveImagePath = dev.outputSaveFolder;
    }
}

