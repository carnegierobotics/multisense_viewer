//
// Created by magnus on 10/12/22.
//

#ifndef MULTISENSE_VIEWER_TRACKINGTEST_H
#define MULTISENSE_VIEWER_TRACKINGTEST_H



#include <MultiSense/src/Scripts/Private/ScriptBuilder.h>
#include "MultiSense/src/Renderer/Renderer.h"
#include "MultiSense/src/Features/VisualOdometry.h"
#include "MultiSense/src/ModelLoaders/glTFModel.h"

class TrackingTest: public Base, public RegisteredInFactory<TrackingTest>, glTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    TrackingTest() {
        s_bRegistered;
    }
    ~TrackingTest() = default;
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<TrackingTest>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "TrackingTest"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_DEFAULT;
    std::unique_ptr<VisualOdometry> vo;

    Page previewTab{};
    bool initialized = false;
    uint32_t width = 960, height = 600;

    void *selection = (void *) "0";

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

};


#endif //MULTISENSE_VIEWER_TRACKINGTEST_H
