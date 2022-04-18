//
// Created by magnus on 11/27/21.
//

#ifndef MULTISENSE_BASE_H
#define MULTISENSE_BASE_H

#include <MultiSense/src/tools/Utils.h>
#include <filesystem>
#include <utility>
#include "MultiSense/src/imgui/UISettings.h"
#include "Camera.h"

#define NUM_POINTS 2048 // Changing this also needs to be changed in the vs shader.

typedef enum ScriptType {
    ArDefault,
    ArPointCloud,
    ArDisabled

} ScriptType;


struct UBOMatrix {
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;
};

struct FragShaderParams {
    glm::vec4 lightColor;
    glm::vec4 objectColor;
    glm::vec4 lightPos;
    glm::vec4 viewPos;
};

struct PointCloudParam {
    glm::mat4 kInverse;
    float width;
    float height;
};

struct PointCloudShader{
    glm::vec4 pos[NUM_POINTS];
    glm::vec4 col[NUM_POINTS];
};



class Base {
public:

    /**@brief A standard set of uniform buffers */
    struct UniformBufferSet {
        Buffer bufferOne;
        Buffer bufferTwo;
        Buffer bufferThree;
    };

    void *bufferOneData;
    void *bufferTwoData;
    void *bufferThreeData;

    struct RenderUtils {
        VulkanDevice *device{};
        UISettings *ui{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};
        std::vector<VkPipelineShaderStageCreateInfo> shaders;
        std::vector<UniformBufferSet> uniformBuffers;

    } renderUtils;

    struct Render {
        uint32_t index;
        const Camera *camera;
        float deltaT;
        float runTime;
    } renderData;


    virtual ~Base() = default;

    /**@brief Pure virtual function called once every frame*/
    virtual void update() = 0;

    /**@brief Pure virtual function called only once when VK is ready to render*/
    virtual void setup() = 0;

    /**@brief Pure virtual function called on every UI update, also each frame*/
    virtual void onUIUpdate(UISettings *uiSettings) = 0;

    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    virtual ScriptType getType() { return type; }

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i) {};

    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    void updateUniformBufferData(Base::Render d, ScriptType scriptType) {
        this->renderData = d;

        update();
        // If initialized
        if (renderUtils.uniformBuffers.empty())
            return;

        UniformBufferSet currentUB = renderUtils.uniformBuffers[renderData.index];
        if (scriptType == ArDefault || scriptType == ArPointCloud) {
            // TODO unceesarry mapping and unmapping occurring here.
            currentUB.bufferOne.map();
            memcpy(currentUB.bufferOne.mapped, bufferOneData, sizeof(UBOMatrix));
            currentUB.bufferOne.unmap();

            currentUB.bufferTwo.map();
            memcpy(currentUB.bufferTwo.mapped, bufferTwoData, sizeof(FragShaderParams));
            currentUB.bufferTwo.unmap();

            if (bufferThreeData == nullptr) return;
            currentUB.bufferThree.map();
            memcpy(currentUB.bufferThree.mapped, bufferThreeData, sizeof(PointCloudParam));
            currentUB.bufferThree.unmap();

        } else if(scriptType == ArPointCloud){
            currentUB.bufferOne.map();
            memcpy(currentUB.bufferOne.mapped, bufferOneData, sizeof(UBOMatrix));
            currentUB.bufferOne.unmap();

            currentUB.bufferTwo.map();
            memcpy(currentUB.bufferTwo.mapped, bufferTwoData, sizeof(PointCloudShader));
            currentUB.bufferTwo.unmap();
        }


    }

    void createUniformBuffers(RenderUtils utils, ScriptType scriptType) {
        if (scriptType == ArDisabled)
            return;

        renderUtils = std::move(utils);
        renderUtils.uniformBuffers.resize(renderUtils.UBCount);

        if (scriptType == ArDefault || scriptType == ArPointCloud) {

            bufferOneData = new UBOMatrix();
            bufferTwoData = new FragShaderParams();
            bufferThreeData = new PointCloudParam();

            for (auto &uniformBuffer: renderUtils.uniformBuffers) {

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferOne, sizeof(UBOMatrix));

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferTwo, sizeof(FragShaderParams));

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferThree, sizeof(PointCloudParam));

            }
        }
        else if (scriptType == ArPointCloud) {
            bufferOneData = new UBOMatrix();
            bufferTwoData = new PointCloudShader();

            for (auto &uniformBuffer: renderUtils.uniformBuffers) {

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferOne, sizeof(UBOMatrix));

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferTwo, sizeof(PointCloudShader));
            }
        }

        setup();
    }

    [[nodiscard]] VkPipelineShaderStageCreateInfo
    loadShader(std::string fileName, VkShaderStageFlagBits stage) const {

        // Check if we have .spv extensions. If not then add it.
        std::size_t extension = fileName.find(".spv");
        if (extension == std::string::npos)
            fileName.append(".spv");


        VkPipelineShaderStageCreateInfo shaderStage = {};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = stage;
        shaderStage.module = Utils::loadShader((Utils::getShadersPath() + fileName).c_str(),
                                               renderUtils.device->logicalDevice);
        shaderStage.pName = "main";
        assert(shaderStage.module != VK_NULL_HANDLE);
        // TODO CLEANUP SHADERMODULES WHEN UNUSED
        return shaderStage;
    }

    ScriptType type = ArDisabled;


protected:

};

#endif //MULTISENSE_BASE_H
