//
// Created by magnus on 11/7/23.
//

#ifndef MULTISENSE_VIEWER_STEREOSIM_H
#define MULTISENSE_VIEWER_STEREOSIM_H


#include "Viewer/ModelLoaders/ComputeShader.h"
#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/ImGui/Layer.h"

/** StereoSim class for Scripts. Inherits Base and RegisteredInFactory by default, but can be extended with any class in the ModelLoaders folder to draw a model. See \refitem MultiSenseCamera for an example of drawing models **/
class StereoSim: public VkRender::Base, public VkRender::RegisteredInFactory<StereoSim>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    StereoSim() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of StereoSim **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<StereoSim>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "StereoSim"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    ScriptTypeFlags getType() override { return type; }
    DrawMethod getDrawMethod() override {return drawMethod;}
    void setDrawMethod(DrawMethod _drawMethod) override{ this->drawMethod = _drawMethod; }
    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptTypeFlags type = CRL_SCRIPT_TYPE_SIMULATED_CAMERA;
    DrawMethod drawMethod = CRL_SCRIPT_DRAW;

    void onDestroy() override {
        topLevelData->compute.reset = true;
    }

    ComputeShader computeShader;
    int64_t lastPresentedFrameID = -1;
    std::chrono::steady_clock::time_point lastPresentTime;
    bool enable = false;

    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b);

    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle);
};


#endif //MULTISENSE_VIEWER_STEREOSIM_H
