#pragma once


#include <MultiSense/src/Scripts/ScriptBuilder.h>
#include <MultiSense/src/ModelLoaders/glTFModel.h>
#include <MultiSense/src/imgui/Layer.h>

/** Example class for Scripts. Inherits Base and RegisteredInFactory by default, but can be extended with any class in the ModelLoaders folder **/
class Example: public Base, public RegisteredInFactory<Example>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Example() {
        s_bRegistered;
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of Example **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Example>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "Example"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    ScriptType getType() override {return type;}
    /** @brief update function called once per frame with a const UI reference handle **/
    void onUIUpdate(MultiSense::GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_DISABLED;

    void *selection = (void *) "0";

    /** @brief draw function called once per frame with handle to command buffer for which a draw cmd can be recorded in **/
    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

};
