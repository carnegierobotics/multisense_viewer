//
// Created by magnus on 11/27/21.
//

#ifndef AR_ENGINE_BASE_H
#define AR_ENGINE_BASE_H

#include <MultiSense/src/Renderer/shaderParams.h>
#include <MultiSense/src/tools/Utils.h>
#include <filesystem>
#include "MultiSense/src/imgui/UISettings.h"
#include "Camera.h"

class Base {
public:

    struct SetupVars {
        VulkanDevice *device{};
        UISettings *ui{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};

    } b;

    /**@brief A standard set of uniform buffers */
    struct UniformBufferSet {
        Buffer vert;
        Buffer frag;
        Buffer fragSelect;
    };

    struct Render {
        uint32_t index;
        void *params;
        void *matrix;
        const Camera *camera;
        float deltaT;
        void *selection;
    } data;


    virtual ~Base() = default;

    virtual void update() = 0;

    virtual void setup(SetupVars vars) = 0;

    virtual void onUIUpdate(UISettings uiSettings) = 0;

    virtual std::string getType() { return type; }

    /**@brief Render Commands TODO: REMOVE**/
    virtual void prepareObject() {};

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i) {};

    void updateUniformBufferData(Base::Render renderData) {
        data = renderData;

        update();
        // If initialized
        if (uniformBuffers.empty())
            return;

        UniformBufferSet currentUB = uniformBuffers[data.index];

        auto d = (UBOMatrix *) data.matrix;

        // TODO unceesarry mapping and unmapping occuring here maybe.
        currentUB.vert.map();
        memcpy(currentUB.vert.mapped, data.matrix, sizeof(UBOMatrix));
        currentUB.vert.unmap();

        currentUB.frag.map();
        memcpy(currentUB.frag.mapped, data.params, sizeof(FragShaderParams));
        currentUB.frag.unmap();


        if (data.selection == NULL)
            return;

        currentUB.fragSelect.map();
        char *val = static_cast<char *>(data.selection);
        float f = (float) atoi(val);
        memcpy(currentUB.fragSelect.mapped, &f, sizeof(float));
        currentUB.fragSelect.unmap();

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
        shaderStage.module = Utils::loadShader((Utils::getShadersPath() + fileName).c_str(), b.device->logicalDevice);
        shaderStage.pName = "main";
        assert(shaderStage.module != VK_NULL_HANDLE);
        // TODO CLEANUP SHADERMODULES WHEN UNUSED
        return shaderStage;
    }

    std::string type = "None";


protected:
    std::vector<UniformBufferSet> uniformBuffers;

    void createUniformBuffers() {
        uniformBuffers.resize(b.UBCount);

        for (auto &uniformBuffer: uniformBuffers) {

            b.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &uniformBuffer.vert, sizeof(UBOMatrix));

            b.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &uniformBuffer.frag, sizeof(FragShaderParams));

            b.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &uniformBuffer.fragSelect, sizeof(float));

        }

    }


};

#endif //AR_ENGINE_BASE_H
