//
// Created by magnus on 9/19/22.
//

#ifndef MULTISENSE_VIEWER_TWO_H
#define MULTISENSE_VIEWER_TWO_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/CRLCameraModels.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"

class Two: public VkRender::Base, public VkRender::RegisteredInFactory<Two>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Two() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP    }
    void onDestroy() override{
        stbi_image_free(m_NoDataTex);
        stbi_image_free(m_NoSourceTex);
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Two>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "Two"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}
    /** @brief called after renderer has handled a window resize event **/
    void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override{ this->type = _type; }

    void onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = CRL_SCRIPT_TYPE_DISABLED;

    std::unique_ptr<CRLCameraModels::Model> m_Model;
    std::unique_ptr<CRLCameraModels::Model> m_NoDataModel;
    std::unique_ptr<CRLCameraModels::Model> m_NoSourceModel;
    enum {
        DRAW_NO_SOURCE = 0,
        DRAW_NO_DATA = 1,
        DRAW_MULTISENSE = 2
    } state;

    float up = -1.3f;
    unsigned char* m_NoDataTex{};
    unsigned char* m_NoSourceTex{};
    Page selectedPreviewTab = CRL_TAB_NONE;
    float posY = 0.0f;
    float scaleX = 0.25f;
    float scaleY = 0.25f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    std::string src;
    int16_t remoteHeadIndex = 0;
    CRLCameraResolution res = CRL_RESOLUTION_NONE;
    CRLCameraDataType textureType = CRL_CAMERA_IMAGE_NONE;

    int64_t lastPresentedFrameID = -1;
    std::chrono::steady_clock::time_point lastPresentTime;
    int texWidth = 0, texHeight = 0, texChannels = 0;

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(const VkRender::GuiObjectHandles * handles, const VkRender::Device& element);

    void prepareMultiSenseTexture();
    void prepareDefaultTexture();

};

#endif //MULTISENSE_VIEWER_TWO_H
