//
// Created by magnus on 7/8/22.
//

#ifndef MULTISENSE_VIEWER_RIGHTPREVIEW_H
#define MULTISENSE_VIEWER_RIGHTPREVIEW_H




#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/CRLCameraModels.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>
#include "MultiSense/src/Renderer/Renderer.h"

class RightPreview: public Base, public RegisteredInFactory<RightPreview>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    RightPreview() {
        s_bRegistered;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<RightPreview>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "RightPreview"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(Base::Render r) override;
    /** @brief update function called once per frame **/
    void update() override {};
    /** @brief update function called once per frame **/
    void update(CameraConnection* conn) override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(AR::GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_CRL_CAMERA;

    CRLCameraModels::Model* model;
    float posY = 0.0f;
    float posXMin = 0.0f;
    float posXMax = 0.0f;
    float posYMin = 0.0f;
    float posYMax = 0.0f;
    int prevOrder = 0;
    bool coordinateTransformed = false;
    void *selection = (void *) "0";
    std::string src;
    CameraPlaybackFlags playbackSate;
    Page selectedPreviewTab = TAB_NONE;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(AR::GuiObjectHandles handles, AR::Element element);
};


#endif //MULTISENSE_VIEWER_RIGHTPREVIEW_H
