#pragma once


#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/imgui/UISettings.h>
#include <MultiSense/src/model_loaders/MeshModel.h>
#include <MultiSense/src/crl_camera/CRLVirtualCamera.h>

class MultiSenseCamera: public Base, public RegisteredInFactory<MultiSenseCamera>, MeshModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSenseCamera(){
        s_bRegistered;
    }
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

    void onUIUpdate(UISettings uiSettings) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = FrDisabled;


    CRLVirtualCamera* virtualCamera;

    void generateGridPoints();
    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;


};
