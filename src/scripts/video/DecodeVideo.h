//
// Created by magnus on 6/27/22.
//

#ifndef MULTISENSE_VIEWER_DECODEVIDEO_H
#define MULTISENSE_VIEWER_DECODEVIDEO_H

#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/CRLCameraModels.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>
#include<bits/stdc++.h>
#include<pthread.h>
#include<semaphore.h>

extern "C" {
#include<libavutil/avutil.h>
#include<libavutil/imgutils.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

class DecodeVideo : public Base, public RegisteredInFactory<DecodeVideo>, CRLCameraModels {

public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    DecodeVideo() {
        s_bRegistered;
    }

    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<DecodeVideo>(); }

    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "DecodeVideo"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;

    /** @brief update function called once per frame **/
    void update() override;

    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override { return type; }

    void onUIUpdate(GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = ArDefault;
    CRLCameraModels::Model *model{};

    int width = 0, height = 0;
    AVFrame videoFrame[5];
    int bufferSize = 0;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

    static void* decode(void* arg);
    void saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame);

    void prepareTextureAfterDecode();

    int childProcessDecode();


// Declaration
    int r1, items = 0;

// Semaphore declaration
    sem_t notEmpty, notFull;


    bool drawFrame;
};


#endif //MULTISENSE_VIEWER_DECODEVIDEO_H
