//
// Created by magnus on 11/4/22.
//

#ifndef MULTISENSE_VIEWER_GIZMOS_H
#define MULTISENSE_VIEWER_GIZMOS_H


#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"

class Gizmos : public VkRender::Base, public VkRender::RegisteredInFactory<Gizmos>, GLTFModel {
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Gizmos() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }

    ~Gizmos() = default;

    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Gizmos>(); }

    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "Gizmos"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;

    /** @brief update function called once per frame **/
    void update() override;

    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override { return type; }

    void onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override{ this->type = _type; }

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = CRL_SCRIPT_TYPE_RENDER;
    std::unique_ptr<GLTFModel::Model> m_Model;

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;
};


#endif //MULTISENSE_VIEWER_GIZMOS_H
