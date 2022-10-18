#pragma once


#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include <MultiSense/Src/ModelLoaders/glTFModel.h>
#include "MultiSense/Src/Renderer/Renderer.h"

class MultiSenseCamera: public Base, public RegisteredInFactory<MultiSenseCamera>, glTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSenseCamera() {
        s_bRegistered;
    }
    ~MultiSenseCamera() = default;
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<MultiSenseCamera>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "MultiSenseCamera"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_DISABLED;

    Page previewTab{};

    void *selection = (void *) "0";

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    bool imuEnabled = false;
};
