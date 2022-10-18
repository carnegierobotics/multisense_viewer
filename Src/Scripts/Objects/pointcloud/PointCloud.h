//
// Created by magnus on 4/11/22.
//

#ifndef MULTISENSE_POINTCLOUD_H
#define MULTISENSE_POINTCLOUD_H

#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include <MultiSense/Src/ModelLoaders/CRLCameraModels.h>
#include <MultiSense/Src/imgui/Layer.h>
#include "MultiSense/Src/Renderer/Renderer.h"
#include "MultiSense/Src/CRLCamera/CRLPhysicalCamera.h"

class PointCloud: public Base, public RegisteredInFactory<PointCloud>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    PointCloud() {
        s_bRegistered;
    }

    void onDestroy() override{

    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<PointCloud>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "PointCloud"; }
    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}
    /** @brief UI update function called once per frame **/
    void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) override;
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override{ this->type = _type; }

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_DISABLED;

    std::unique_ptr<CRLCameraModels::Model> model;

    uint32_t remoteHeadIndex = 0;
    std::vector<std::string> startedSources{};
    CameraPlaybackFlags playbackSate{};
    Page selectedPreviewTab = TAB_NONE;
    uint32_t width{}, height{};
    CRLCameraResolution res = CRL_RESOLUTION_NONE;

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    int point = 0;

    void prepareTexture();
};


#endif //MULTISENSE_POINTCLOUD_H
