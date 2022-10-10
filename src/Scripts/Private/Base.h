//
// Created by magnus on 11/27/21.
//

#ifndef MULTISENSE_BASE_H
#define MULTISENSE_BASE_H

#include <filesystem>
#include <utility>
#include "MultiSense/src/Core/Camera.h"
#include "MultiSense/src/Tools/Utils.h"
#include "MultiSense/src/Tools/Logger.h"
#include "GLFW/glfw3.h"
#include "MultiSense/src/Core/KeyInput.h"
#include "MultiSense/src/CRLCamera/CRLPhysicalCamera.h"


class CameraConnection; // forward declaration of this class to speed up compile time. Separate Scripts/model_loaders from ImGui source recompile

class Base {
public:

    std::unique_ptr<VkRender::UBOMatrix> bufferOneData{};
    std::unique_ptr<VkRender::FragShaderParams> bufferTwoData{};
    std::unique_ptr<VkRender::PointCloudParam> bufferThreeData{};
    std::unique_ptr<VkRender::ZoomParam> bufferFourData{};

    std::vector<VkShaderModule> shaderModules{};

    VkRender::RenderUtils renderUtils{};

    struct Render {
        uint32_t index;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        std::unique_ptr<CRLPhysicalCamera> *crlCamera;
        ScriptType type;
        Log::Logger *pLogger;
        uint32_t height;
        uint32_t width;
        const Input *input;
    } renderData{};


    virtual ~Base() = default;

    /**@brief Pure virtual function called only once when VK is ready to render*/
    virtual void setup() {
        if (getType() != AR_SCRIPT_TYPE_DISABLED)
            Log::Logger::getInstance()->info("Function setup not overridden for {} script", renderData.scriptName);
    };
    /**@brief Pure virtual function called once every frame*/
    virtual void update() {
        if (getType() != AR_SCRIPT_TYPE_DISABLED)
            Log::Logger::getInstance()->info("Function update not overridden for {} script", renderData.scriptName);
    };
    /**@brief Pure virtual function called each frame*/
    virtual void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
        if (getType() != AR_SCRIPT_TYPE_DISABLED)
            Log::Logger::getInstance()->info("Function onUIUpdate not overridden for {} script", renderData.scriptName);
    };
    /**@brief Pure virtual function called to enable/disable drawing of this script*/
    virtual void setDrawMethod(ScriptType type) {
        if (getType() != AR_SCRIPT_TYPE_DISABLED)
            Log::Logger::getInstance()->info("Function setDrawMethod not overridden for {} script", renderData.scriptName);
    };
    /**@brief Virtual function called when resize event is triggered from the platform os*/
    virtual void onWindowResize(const MultiSense::GuiObjectHandles *uiHandle) {
        if (getType() != AR_SCRIPT_TYPE_DISABLED)
            Log::Logger::getInstance()->info("Function onWindowResize not overridden for {} script", renderData.scriptName);
    };
    /**@brief Called once script is requested for deletion */
    virtual void onDestroy() {
        if (getType() != AR_SCRIPT_TYPE_DISABLED)
            Log::Logger::getInstance()->info("Function onDestroy not overridden for {} script", renderData.scriptName);

    };
    /**@brief Which script type this is. Can be used to flashing/disable rendering of this script */
    virtual ScriptType getType() {
        return AR_SCRIPT_TYPE_DISABLED;
    }


    void windowResize(Render *data, const MultiSense::GuiObjectHandles *uiHandle) {
        updateRenderData(data);

        onWindowResize(uiHandle);
    }



    void uiUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
        if (!this->renderData.drawThisScript)
            return;

        if (renderData.crlCamera != nullptr)
            onUIUpdate(uiHandle);

    }

    void drawScript(VkCommandBuffer commandBuffer, uint32_t i, bool b) {

        if (!renderData.drawThisScript)
            return;

        auto time = std::chrono::steady_clock::now();
        std::chrono::duration<float> time_span =
                std::chrono::duration_cast<std::chrono::duration<float>>(time - lastLogTime);
        if (time_span.count() > INTERVAL_10_SECONDS_LOG_DRAW_COUNT || renderData.scriptDrawCount == 0) {
            lastLogTime = std::chrono::steady_clock::now();
            renderData.pLogger->info("Draw-number: {} for script: {}", (int) renderData.scriptDrawCount,
                                     renderData.scriptName.c_str());
        }
        draw(commandBuffer, i, b);
        if (i == 0)
            renderData.scriptDrawCount++;

    };

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
        Log::Logger::getInstance()->info("draw not overridden for {} script", renderData.scriptName);

    };

    void updateUniformBufferData(Render *data) {
        updateRenderData(data);

        renderData.scriptRuntime = (float) (std::chrono::steady_clock::now() - startTime).count();

        if (renderData.crlCamera->get() != nullptr)
            update();

        VkRender::UniformBufferSet &currentUB = renderUtils.uniformBuffers[renderData.index];
        if (renderData.type != AR_SCRIPT_TYPE_DISABLED) {
            memcpy(currentUB.bufferOne.mapped, bufferOneData.get(), sizeof(VkRender::UBOMatrix));
            memcpy(currentUB.bufferTwo.mapped, bufferTwoData.get(), sizeof(VkRender::FragShaderParams));
            memcpy(currentUB.bufferThree.mapped, bufferThreeData.get(), sizeof(VkRender::PointCloudParam));
            memcpy(currentUB.bufferFour.mapped, bufferFourData.get(), sizeof(VkRender::ZoomParam));
        }
    }

    void createUniformBuffers(const VkRender::RenderUtils &utils, Base::Render rData) {
        renderData = std::move(rData);
        renderUtils = utils;
        renderUtils.uniformBuffers.resize(renderUtils.UBCount);
        startTime = std::chrono::steady_clock::now();
        lastLogTime = std::chrono::steady_clock::now();
        bufferOneData = std::make_unique<VkRender::UBOMatrix>();
        bufferTwoData = std::make_unique<VkRender::FragShaderParams>();
        bufferThreeData = std::make_unique<VkRender::PointCloudParam>();
        bufferFourData = std::make_unique<VkRender::ZoomParam>();
        for (auto &uniformBuffer: renderUtils.uniformBuffers) {
            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             &uniformBuffer.bufferOne, sizeof(VkRender::UBOMatrix));
            uniformBuffer.bufferOne.map();

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             &uniformBuffer.bufferTwo, sizeof(VkRender::FragShaderParams));
            uniformBuffer.bufferTwo.map();

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             &uniformBuffer.bufferThree, sizeof(VkRender::PointCloudParam));
            uniformBuffer.bufferThree.map();

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             &uniformBuffer.bufferFour, sizeof(VkRender::ZoomParam));
            uniformBuffer.bufferFour.map();

        }
        renderData.scriptRuntime = (float) (std::chrono::steady_clock::now() - startTime).count();
        setup();
        renderData.drawThisScript = true;
    }

    /**@brief Call to delete the attached script. */
    void onDestroyScript() {
        for (auto *shaderModule: shaderModules) {
            vkDestroyShaderModule(renderUtils.device->logicalDevice, shaderModule, nullptr);
        }

        for (auto &uniformBuffer: renderUtils.uniformBuffers) {
            uniformBuffer.bufferOne.unmap();
            uniformBuffer.bufferTwo.unmap();
            uniformBuffer.bufferThree.unmap();
            uniformBuffer.bufferFour.unmap();
        }

        onDestroy();
    }

    [[nodiscard]] VkPipelineShaderStageCreateInfo
    loadShader(std::string fileName, VkShaderStageFlagBits stage) {

        // Check if we have .spv extensions. If not then add it.
        std::size_t extension = fileName.find(".spv");
        if (extension == std::string::npos)
            fileName.append(".spv");

        VkShaderModule module;
        Utils::loadShader((Utils::getShadersPath() + fileName).c_str(),
                          renderUtils.device->logicalDevice, &module);
        assert(module != VK_NULL_HANDLE);

        shaderModules.emplace_back(module);
        VkPipelineShaderStageCreateInfo shaderStage = {};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = stage;
        shaderStage.module = module;
        shaderStage.pName = "main";


        return shaderStage;
    }

protected:
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime;
    std::chrono::steady_clock::time_point lastLogTime;


    void updateRenderData(Render *data) {
        this->renderData.camera = data->camera;
        this->renderData.crlCamera = data->crlCamera;
        this->renderData.deltaT = data->deltaT;
        this->renderData.index = data->index;
        this->renderData.pLogger = data->pLogger;
        this->renderData.height = data->height;
        this->renderData.width = data->width;
        this->renderData.type = getType();
        input = data->input;
    }

    const Input *input{};
};

#endif //MULTISENSE_BASE_H