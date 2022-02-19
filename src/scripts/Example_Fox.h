#pragma once


#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/imgui/UISettings.h>
#include <MultiSense/src/core/glTFModel.h>

class Example_Fox: public Base, public RegisteredInFactory<Example_Fox>, glTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Example_Fox() {
        s_bRegistered;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Example_Fox>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "Example_Fox"; }

    /** @brief Setup function called one during engine prepare **/
    void setup(SetupVars vars) override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    std::string getType() override;

    void generateSquare();
    void onUIUpdate(UISettings uiSettings) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    std::string type = "Render";

    void *selection = (void *) "0";

    void prepareObject() override;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

};
