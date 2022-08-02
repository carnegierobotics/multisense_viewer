//
// Created by magnus on 6/27/22.
//

#ifndef MULTISENSE_VIEWER_DECODEVIDEO_H
#define MULTISENSE_VIEWER_DECODEVIDEO_H

#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/CRLCameraModels.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>


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
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(CameraConnection* camHandle) override;

    /** @brief update function called once per frame **/
    void update() override;

    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override { return type; }

    void onUIUpdate(GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = ArCameraScript;
    CRLCameraModels::Model *model{};

    int width = 0, height = 0;
    std::string src;
    uint32_t playbackSate = 1;
    CameraConnection* camHandle;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

    static void *decode(void *arg);

    void saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame);

    void prepareTextureAfterDecode();

    int childProcessDecode();
};


#endif //MULTISENSE_VIEWER_DECODEVIDEO_H
