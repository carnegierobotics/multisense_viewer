//
// Created by magnus on 2/25/23.
//

#ifndef MULTISENSE_VIEWER_SKYBOX_H
#define MULTISENSE_VIEWER_SKYBOX_H



#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"

class Skybox : public VkRender::Base, public VkRender::RegisteredInFactory<Skybox>, GLTFModel {
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Skybox() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }

    ~Skybox(){
        vkDestroyImageView(renderUtils.device->m_LogicalDevice, skyboxTextures.irradianceCube.m_View, nullptr);
        vkDestroyImage(renderUtils.device->m_LogicalDevice, skyboxTextures.irradianceCube.m_Image, nullptr);
        vkDestroySampler(renderUtils.device->m_LogicalDevice, skyboxTextures.irradianceCube.m_Sampler, nullptr);
        vkFreeMemory(renderUtils.device->m_LogicalDevice, skyboxTextures.irradianceCube.m_DeviceMemory, nullptr);

        vkDestroyImageView(renderUtils.device->m_LogicalDevice, skyboxTextures.prefilterEnv.m_View, nullptr);
        vkDestroyImage(renderUtils.device->m_LogicalDevice, skyboxTextures.prefilterEnv.m_Image, nullptr);
        vkDestroySampler(renderUtils.device->m_LogicalDevice, skyboxTextures.prefilterEnv.m_Sampler, nullptr);
        vkFreeMemory(renderUtils.device->m_LogicalDevice, skyboxTextures.prefilterEnv.m_DeviceMemory, nullptr);

        vkDestroyImageView(renderUtils.device->m_LogicalDevice, skyboxTextures.lutBrdf.m_View, nullptr);
        vkDestroyImage(renderUtils.device->m_LogicalDevice, skyboxTextures.lutBrdf.m_Image, nullptr);
        vkDestroySampler(renderUtils.device->m_LogicalDevice, skyboxTextures.lutBrdf.m_Sampler, nullptr);
        vkFreeMemory(renderUtils.device->m_LogicalDevice, skyboxTextures.lutBrdf.m_DeviceMemory, nullptr);
    }

    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Skybox>(); }

    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "Skybox"; }

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
    ScriptType type = CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE;
    std::unique_ptr<GLTFModel::Model> skybox;
    float exposure = 4.5f;
    float gamma = 2.2f;
    float ibl = 1.0f;
    int debugViewInputs = 0;
    float lod = 1.5f;

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;
};


#endif //MULTISENSE_VIEWER_SKYBOX_H
