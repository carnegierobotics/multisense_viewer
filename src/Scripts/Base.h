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

    /**@brief A standard set of uniform buffers */
    struct UniformBufferSet {
        Buffer bufferOne;
        Buffer bufferTwo;
        Buffer bufferThree;
        Buffer bufferFour;
    };

    std::unique_ptr<VkRender::UBOMatrix> bufferOneData{};
    std::unique_ptr<VkRender::FragShaderParams> bufferTwoData{};
    std::unique_ptr<VkRender::PointCloudParam> bufferThreeData{};
    std::unique_ptr<VkRender::ZoomParam> bufferFourData{};

    std::vector<VkShaderModule> shaderModules{};

    struct RenderUtils {
        VulkanDevice* device{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};
        std::vector<UniformBufferSet> uniformBuffers;
        const VkRender::ObjectPicking* picking;
    } renderUtils{};

    struct Render {
        uint32_t index;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        std::unique_ptr<CRLPhysicalCamera>* crlCamera;
        ScriptType type;
        Log::Logger *pLogger;
        uint32_t height;
        uint32_t width;
        const Input *input;
    } renderData{};


    virtual ~Base() = default;

    /**@brief Pure virtual function called once every frame*/
    virtual void update() {};

    /**@brief Pure virtual function called only once when VK is ready to render*/
    virtual void setup() {};

    /**@brief Virtual function called once when VK is ready to render with camera handle
     * @param camHandle: Handle to currently connected camera
     * Called if the script type is: ArCameraScript and
     * the cameraHandle has been initialized by the CameraConnection Class */
    virtual void setup(Render r) {};

    virtual void onWindowResize(const MultiSense::GuiObjectHandles *uiHandle) {

    };

    void windowResize(Render *data, const MultiSense::GuiObjectHandles* uiHandle) {
        updateRenderData(data);

        onWindowResize(uiHandle);
    }

    /**@brief Pure virtual function called on every UI update, also each frame*/
    virtual void onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {};

    void uiUpdate(const MultiSense::GuiObjectHandles* uiHandle) {
        if (!this->renderData.drawThisScript)
            return;

        if (renderData.crlCamera != NULL)
            onUIUpdate(uiHandle);

    }


    /**@brief Which script type this is. Can be used to flashing/disable rendering of this script */
    virtual ScriptType getType() { return AR_SCRIPT_TYPE_DISABLED; }

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

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {};

    /**@brief Which script type this is. Can be used to flashing/disable rendering of this script */
    void updateUniformBufferData(Render *data) {
        updateRenderData(data);

        renderData.scriptRuntime = (float) (std::chrono::steady_clock::now() - startTime).count();

        if (renderData.crlCamera->get() != nullptr)
            update();

        UniformBufferSet& currentUB = renderUtils.uniformBuffers[renderData.index];
        if (renderData.type != AR_SCRIPT_TYPE_DISABLED) {
            memcpy(currentUB.bufferOne.mapped, bufferOneData.get(), sizeof(VkRender::UBOMatrix));
            memcpy(currentUB.bufferTwo.mapped, bufferTwoData.get(), sizeof(VkRender::FragShaderParams));
            memcpy(currentUB.bufferThree.mapped, bufferThreeData.get(), sizeof(VkRender::PointCloudParam));
            memcpy(currentUB.bufferFour.mapped, bufferFourData.get(), sizeof(VkRender::ZoomParam));
        }
    }

    void createUniformBuffers(const RenderUtils &utils, Base::Render rData) {
        if (this->getType() == AR_SCRIPT_TYPE_DISABLED)
            return;
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
        if (getType() == AR_SCRIPT_TYPE_CRL_CAMERA || getType() == AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY ||
            getType() == AR_SCRIPT_TYPE_POINT_CLOUD)
            setup(renderData);
        else
            setup();

        renderData.drawThisScript = true;
    }


    /**@brief Called once script is deleted */
    virtual void onDestroy() {};

    /**@brief Call to delete the attached script. */
    void onDestroyScript() {
        for (auto * shaderModule: shaderModules) {
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
