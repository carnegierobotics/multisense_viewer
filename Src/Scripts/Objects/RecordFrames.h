//
// Created by magnus on 10/12/22.
//

#ifndef MULTISENSE_VIEWER_RECORDFRAMES_H
#define MULTISENSE_VIEWER_RECORDFRAMES_H


#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include <MultiSense/Src/ModelLoaders/CRLCameraModels.h>
#include <MultiSense/Src/imgui/Layer.h>
#include "MultiSense/Src/Renderer/Renderer.h"
#include "MultiSense/Src/CRLCamera/CRLPhysicalCamera.h"

class RecordFrames: public Base, public RegisteredInFactory<RecordFrames>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    RecordFrames() {
        s_bRegistered;
    }
    void onDestroy() override{
        for (const auto& buf : dataBufferMap){
            free(buf.second);
        }
        for (const auto& buf : dataBufferMapSecondary){
            free(buf.second);
        }
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<RecordFrames>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "RecordFrames"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override{ this->type = _type; }

    void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) override;
    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override {}

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_DEFAULT;

    std::unique_ptr<ThreadPool> threadPool;

    bool saveImage = false;
    std::string saveImagePath;
    std::vector<std::string> sources;
    std::unordered_map<std::string, uint32_t> ids;
    int16_t remoteHeadIndex = 0;
    CRLCameraDataType textureType = AR_CAMERA_IMAGE_NONE;

    std::unordered_map<std::string, uint8_t*> dataBufferMap;
    std::unordered_map<std::string, uint8_t*> dataBufferMapSecondary;

    bool initializeBuffers = false;
};





#endif //MULTISENSE_VIEWER_RECORDFRAMES_H
