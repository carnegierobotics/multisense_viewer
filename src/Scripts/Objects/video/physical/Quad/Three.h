//
// Created by magnus on 9/19/22.
//

#ifndef MULTISENSE_VIEWER_THREE_H
#define MULTISENSE_VIEWER_THREE_H


#include <MultiSense/src/Scripts/Private/ScriptBuilder.h>
#include <MultiSense/src/ModelLoaders/CRLCameraModels.h>
#include <MultiSense/src/imgui/Layer.h>
#include "MultiSense/src/Renderer/Renderer.h"
#include "MultiSense/src/CRLCamera/CRLPhysicalCamera.h"
class Three: public Base, public RegisteredInFactory<Three>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Three() {
        s_bRegistered;
    }
    void onDestroy() override{
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Three>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "Three"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}
    /** @brief called after renderer has handled a window resize event **/
    void onWindowResize(const MultiSense::GuiObjectHandles *uiHandle) override;
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override{ this->type = _type; }

    void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_DISABLED;

    std::unique_ptr<CRLCameraModels::Model> model;

    int count = 1;
    void *selection = (void *) "0";
    float up = -1.3f;
    bool coordinateTransformed = false;
    Page selectedPreviewTab = TAB_NONE;
    float posY = 0.0f;
    float scaleX = 0.25f;
    float scaleY = 0.25f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    std::string src;
    uint32_t remoteHeadIndex = 0;
    CRLCameraResolution res = CRL_RESOLUTION_NONE;
    CameraPlaybackFlags playbackSate{};
    uint32_t width = 0, height = 0;
    CRLCameraDataType textureType = AR_CAMERA_IMAGE_NONE;

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(const MultiSense::GuiObjectHandles * handles, MultiSense::Device element);

    void prepareTexture();
};


#endif //MULTISENSE_VIEWER_THREE_H
