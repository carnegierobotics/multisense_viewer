//
// Created by magnus on 3/10/22.
//

#ifndef MULTISENSE_DEFAULTPREVIEW_H
#define MULTISENSE_DEFAULTPREVIEW_H


#include "ScriptHeader.h"


class DefaultPreview: public Base, public RegisteredInFactory<DefaultPreview>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    DefaultPreview() {
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
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<DefaultPreview>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "DefaultPreview"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(Base::Render renderData) override;
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

    CRLCameraModels::Model* model{}; // TODO DELETE POINTER IN CLEANUP

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
    std::vector<std::string> startedSources;
    uint32_t width, height;

    CameraPlaybackFlags playbackSate;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(AR::GuiObjectHandles handles, AR::Element element);

};


#endif //MULTISENSE_DEFAULTPREVIEW_H
