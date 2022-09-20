//
// Created by magnus on 6/27/22.
//

#ifndef MULTISENSE_VIEWER_LEFTIMAGER_H
#define MULTISENSE_VIEWER_LEFTIMAGER_H

#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/CRLCameraModels.h>
#include <MultiSense/src/imgui/Layer.h>
#include "MultiSense/src/Renderer/Renderer.h"


class LeftImager : public Base, public RegisteredInFactory<LeftImager>, CRLCameraModels {

public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    LeftImager() {
        s_bRegistered;
    }

    void onDestroy() override{
        delete model;
        //camHandle->camPtr->stop(AR_PREVIEW_VIRTUAL_RIGHT);
    }


    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<LeftImager>(); }

    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "LeftImager"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(Base::Render r) override;

    /** @brief update function called once per frame **/
    void update() override;

    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override { return type; }
    /** @brief called after renderer has handled a window resize event **/
    void onWindowResize(AR::GuiObjectHandles uiHandle) override;


    void onUIUpdate(AR::GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY;
    CRLCameraModels::Model *model{};

    uint32_t width = 0, height = 0;
    std::string src;
    CameraPlaybackFlags playbackSate;
    CameraConnection* camHandle{};
    float posY = 0.0f;
    float scaleX = 0.25f;
    float scaleY = 0.25f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float posXMin = 0.0f;
    float posXMax = 0.0f;
    float posYMin = 0.0f;
    float posYMax = 0.0f;
    float speed = 1.0f;
    int prevOrder = 0;

    std::chrono::steady_clock::time_point start, end;

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    static void *decode(void *arg);


    void prepareTextureAfterDecode();

    int childProcessDecode();

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(AR::GuiObjectHandles handles, AR::Element element);
};


#endif //MULTISENSE_VIEWER_LEFTIMAGER_H
