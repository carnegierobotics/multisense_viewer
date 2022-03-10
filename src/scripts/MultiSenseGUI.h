//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_MULTISENSEGUI_H
#define MULTISENSE_MULTISENSEGUI_H

#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/imgui/UISettings.h>
#include <MultiSense/src/model_loaders/MeshModel.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>

/**
 * MultiSenseGUI. This script generates most of the necessary GUI elements.
 */
class MultiSenseGUI: public Base, public RegisteredInFactory<MultiSenseGUI>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSenseGUI(){
        s_bRegistered;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<MultiSenseGUI>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "MultiSenseGUI"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(UISettings *uiSettings) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = ArDefault;

    CRLPhysicalCamera* camera;
    Button* connectButton;
    Text* cameraNameHeader;
    Text* cameraName;
    DropDownItem *modes;

    Button* startStream;



    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;


};

#endif //MULTISENSE_MULTISENSEGUI_H
