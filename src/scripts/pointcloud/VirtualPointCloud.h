#pragma once


#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/MeshModel.h>
#include "MultiSense/src/model_loaders/CRLCameraModels.h"
#include "MultiSense/src/Renderer/Renderer.h"


class VirtualPointCloud: public Base, public RegisteredInFactory<VirtualPointCloud>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    VirtualPointCloud(){
        s_bRegistered;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<VirtualPointCloud>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "VirtualPointCloud"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(Base::Render r) override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(AR::GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_POINT_CLOUD;

    CRLCameraModels::Model* model{};
    CameraPlaybackFlags playbackSate{};
    Page selectedPreviewTab = TAB_NONE;

    const int vertexCount = 960 * 600;
    ArEngine::Vertex *meshData{}; // Don't forget to de

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;


};
