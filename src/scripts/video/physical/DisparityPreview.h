//
// Created by magnus on 5/8/22.
//

#ifndef MULTISENSE_PREVIEWBAR_H
#define MULTISENSE_PREVIEWBAR_H


#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/CRLCameraModels.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>
#include "MultiSense/src/Renderer/Renderer.h"

class DisparityPreview: public Base, public RegisteredInFactory<DisparityPreview>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    DisparityPreview() {
        s_bRegistered;
    }
    void onDestroy() override{
        delete model;
        for(const auto& source : startedSources){
            auto* ptr = dynamic_cast<CRLPhysicalCamera *>(renderData.crlCamera->get()->camPtr);
            ptr->stop(source);
        }
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<DisparityPreview>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "DisparityPreview"; }

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
    /** @brief called after renderer has handled a window resize event **/
    void onWindowResize(AR::GuiObjectHandles uiHandle) override;


    void onUIUpdate(AR::GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_CRL_CAMERA;

    CRLCameraModels::Model* model;

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
    float posXMin = 0.0f;
    float posXMax = 0.0f;
    float posYMin = 0.0f;
    float posYMax = 0.0f;
    float speed = 1.0f;
    int prevOrder = 0;
    std::string src = "source";
    std::string resolution = "resolution";
    std::vector<std::string> startedSources;
    CameraPlaybackFlags playbackSate{};
    float aspectRatio = 1.0f;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(AR::GuiObjectHandles handles, AR::Element element);
};

#endif //MULTISENSE_PREVIEWBAR_H
