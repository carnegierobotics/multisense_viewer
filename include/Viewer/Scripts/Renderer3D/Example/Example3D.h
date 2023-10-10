//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_EXAMPLE_3D_H
#define MULTISENSE_VIEWER_EXAMPLE_3D_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"


class Example3D: public VkRender::Base, public VkRender::RegisteredInFactory<Example3D>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Example3D() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of Example3D **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Example3D>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "Example3D"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
        KS21.reset();
    }
    /** @brief set if this script should be drawn or not. */
    void setDrawMethod(DrawMethod _drawMethod) override{ this->drawMethod = _drawMethod; }

    /** @brief draw function called once per frame **/
    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    ScriptTypeFlags getType() override { return type; }
    DrawMethod getDrawMethod() override {return drawMethod;}

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptTypeFlags type = CRL_SCRIPT_TYPE_DISABLED;
    DrawMethod drawMethod = CRL_SCRIPT_DONT_DRAW;

    std::unique_ptr<GLTFModel::Model> KS21;
    struct LightSource {
        glm::vec3 color = glm::vec3(1.0f);
        glm::vec3 rotation = glm::vec3(75.0f, 40.0f, 0.0f);
    } lightSource;
};


#endif //MULTISENSE_VIEWER_EXAMPLE_3D_H
