//
// Created by magnus on 3/10/22.
//

#ifndef MULTISENSE_DEFAULTPREVIEW_H
#define MULTISENSE_DEFAULTPREVIEW_H


#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/model_loaders/CRLCameraModels.h>
#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>

class DefaultPreview: public Base, public RegisteredInFactory<DefaultPreview>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    DefaultPreview() {
        s_bRegistered;
    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<DefaultPreview>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "DefaultPreview"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override {};
    /** @brief Setup function called one during engine prepare **/
    void setup(Base::Render r) override;
    /** @brief update function called once per frame **/
    void update() override {};
    /** @brief update function called once per frame **/
    void update(CameraConnection* conn) override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(GuiObjectHandles uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = ArCameraScript;

    CRLCameraModels::Model* model;

    int count = 1;
    void *selection = (void *) "0";
    float up = -1.3f;

    std::string src;
    uint32_t playbackSate = -1;

    void draw(VkCommandBuffer commandBuffer, uint32_t i) override;

    float posX, posY, posZ;

};



#endif //MULTISENSE_DEFAULTPREVIEW_H
