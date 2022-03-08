//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_MULTISENSES30_H
#define MULTISENSE_MULTISENSES30_H

#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/imgui/UISettings.h>
#include <MultiSense/src/model_loaders/MeshModel.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>

class MultiSenseS30: public Base, public RegisteredInFactory<MultiSenseS30>, MeshModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSenseS30(){
        s_bRegistered;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<MultiSenseS30>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "MultiSenseS30"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(UISettings uiSettings) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = FrDefault;
    CRLPhysicalCamera* camera;
    MeshModel::Model* model;
    int count = 1;

    Button* connectButton;
    Text* cameraNameHeader;
    Text* cameraName;
    DropDownItem *modes;

    Button* startStream;



    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;


};

#endif //MULTISENSE_MULTISENSES30_H
