//
// Created by magnus on 11/27/21.
//

#ifndef AR_ENGINE_BASE_H
#define AR_ENGINE_BASE_H

#include <MultiSense/src/Renderer/shaderParams.h>
#include <MultiSense/src/tools/Utils.h>
#include <filesystem>
#include <utility>
#include "MultiSense/src/imgui/UISettings.h"
#include "Camera.h"

class Base {
public:

    /**@brief A standard set of uniform buffers */
    struct UniformBufferSet {
        Buffer vert;
        Buffer frag;
        Buffer fragSelect;
    };

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
        void *params;
        void *matrix;
        const Camera *camera;
        float deltaT;
        void *selection;
    } renderData;


    virtual ~Base() = default;

    /**@brief Pure virtual function called once every frame*/
    virtual void update() = 0;

    /**@brief Pure virtual function called only once when VK is ready to render*/
    virtual void setup() = 0;

    /**@brief Pure virtual function called on every UI update, also each frame*/
    virtual void onUIUpdate(UISettings uiSettings) = 0;

    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    virtual std::string getType() { return type; }

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i) {};

    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    void updateUniformBufferData(Base::Render d) {
        this->renderData = d;

        update();
        // If initialized
        if (renderUtils.uniformBuffers.empty())
            return;

        UniformBufferSet currentUB = renderUtils.uniformBuffers[renderData.index];

        // TODO unceesarry mapping and unmapping occurring here.
        currentUB.vert.map();
        memcpy(currentUB.vert.mapped, renderData.matrix, sizeof(UBOMatrix));
        currentUB.vert.unmap();

        currentUB.frag.map();
        memcpy(currentUB.frag.mapped, renderData.params, sizeof(FragShaderParams));
        currentUB.frag.unmap();


        if (renderData.selection == NULL)
            return;

        currentUB.fragSelect.map();
        char *val = static_cast<char *>(renderData.selection);
        float f = (float) atoi(val);
        memcpy(currentUB.fragSelect.mapped, &f, sizeof(float));
        currentUB.fragSelect.unmap();

    }

    void createUniformBuffers(RenderUtils utils) {
        renderUtils = std::move(utils);
        renderUtils.uniformBuffers.resize(renderUtils.UBCount);

        for (auto &uniformBuffer: renderUtils.uniformBuffers) {

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &uniformBuffer.vert, sizeof(UBOMatrix));

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &uniformBuffer.frag, sizeof(FragShaderParams));

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &uniformBuffer.fragSelect, sizeof(float));

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
        shaderStage.module = Utils::loadShader((Utils::getShadersPath() + fileName).c_str(), renderUtils.device->logicalDevice);
        shaderStage.pName = "main";
        assert(shaderStage.module != VK_NULL_HANDLE);
        // TODO CLEANUP SHADERMODULES WHEN UNUSED
        return shaderStage;
    }

    std::string type = "None";


protected:

};

#endif //AR_ENGINE_BASE_H
