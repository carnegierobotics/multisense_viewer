//
// Created by magnus on 4/11/22.
//

#ifndef MULTISENSE_POINTCLOUD_H
#define MULTISENSE_POINTCLOUD_H

#include <MultiSense/src/Scripts/ScriptBuilder.h>
#include <MultiSense/src/ModelLoaders/CRLCameraModels.h>
#include <MultiSense/src/imgui/Layer.h>
#include "MultiSense/src/Renderer/Renderer.h"
#include "MultiSense/src/CRLCamera/CRLPhysicalCamera.h"

class PointCloud: public Base, public RegisteredInFactory<PointCloud>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    PointCloud() {
        s_bRegistered;
    }

    void onDestroy() override{
        delete model;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<PointCloud>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "PointCloud"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(Base::Render r) override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief update function called once per frame **/
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_POINT_CLOUD;

    CRLCameraModels::Model* model{};

    std::string src;
    std::vector<std::string> startedSources{};
    CameraPlaybackFlags playbackSate{};
    Page selectedPreviewTab = TAB_NONE;
    uint32_t width{}, height{};
    CRLCameraResolution res = CRL_RESOLUTION_NONE;
    CRLCameraDataType textureType{};

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    VkRender::Vertex* meshData{};
    int point = 0;

    void prepareTexture();
};


#endif //MULTISENSE_POINTCLOUD_H
