//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_MULTISENSE_H
#define MULTISENSE_VIEWER_MULTISENSE_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/ModelLoaders/CustomModels.h"

class MultiSense: public VkRender::Base, public VkRender::RegisteredInFactory<MultiSense>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSense() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of MultiSense **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<MultiSense>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "MultiSense"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
        KS21.reset();
    }
    /** @brief set if this script should be drawn or not. */
    void setDrawMethod(VkRender::CRL_SCRIPT_DRAW_METHOD _drawMethod) override{ this->drawMethod = _drawMethod; }

    /** @brief draw function called once per frame **/
    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b) override;

    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    VkRender::ScriptTypeFlags getType() override { return type; }
    VkRender::CRL_SCRIPT_DRAW_METHOD getDrawMethod() override {return drawMethod;}

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    VkRender::ScriptTypeFlags type = VkRender::CRL_SCRIPT_TYPE_RENDERER3D;
    VkRender::CRL_SCRIPT_DRAW_METHOD drawMethod = VkRender::CRL_SCRIPT_DONT_DRAW;

    std::unique_ptr<GLTFModel::Model> KS21;

    struct LightSource {
        glm::vec3 color = glm::vec3(1.0f);
        glm::vec3 rotation = glm::vec3(75.0f, 40.0f, 0.0f);
    } lightSource;

    std::chrono::system_clock::time_point time;

    std::chrono::steady_clock::time_point lastPrintedTime;
    std::chrono::steady_clock::time_point startPlay;
    size_t entryIdx = 0;


};


#endif //MULTISENSE_VIEWER_MULTISENSE_H
