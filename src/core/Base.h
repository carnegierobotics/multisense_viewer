//
// Created by magnus on 11/27/21.
//

#ifndef MULTISENSE_BASE_H
#define MULTISENSE_BASE_H

#include <filesystem>
#include <utility>
#include "Camera.h"
#include "CameraConnection.h"
#include "MultiSense/src/tools/Utils.h"
#include "MultiSense/src/tools/Logger.h"

#define NUM_POINTS 2048 // Changing this also needs to be changed in the vs shader.

typedef enum ScriptType {
    AR_SCRIPT_TYPE_DISABLED,
    AR_SCRIPT_TYPE_DEFAULT,
    AR_SCRIPT_TYPE_CRL_CAMERA,
    AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY,
    AR_SCRIPT_TYPE_POINT_CLOUD
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

struct PointCloudShader {
    glm::vec4 pos[NUM_POINTS];
    glm::vec4 col[NUM_POINTS];
};

// PreDeclare
struct GuiObjectHandles;

class Base {
public:

    /**@brief A standard set of uniform buffers */
    struct UniformBufferSet {
        Buffer bufferOne;
        Buffer bufferTwo;
        Buffer bufferThree;
    };

    // TODO DELETE POINTERS AS WELL
    void *bufferOneData{};
    void *bufferTwoData{};
    void *bufferThreeData{};

    struct RenderUtils {
        VulkanDevice *device{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};
        std::vector<VkPipelineShaderStageCreateInfo> shaders;
        std::vector<UniformBufferSet> uniformBuffers;

    } renderUtils;

    struct Render {
        uint32_t index;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool finishedSetup = false;
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        std::unique_ptr<CameraConnection> *crlCamera = nullptr;
        ScriptType type;
        std::vector<AR::Element> gui;
        Log::Logger *pLogger;

    } renderData{};


    virtual ~Base() = default;

    /**@brief Pure virtual function called once every frame*/
    virtual void update() = 0;

    /**@brief Optional virtual function usefully for camera scripts
     * Called if the script type is: ArCameraScript and
     * the cameraHandle has been initialized by the CameraConnection Class */
    // TODO Refactor and remove this
    virtual void update(CameraConnection *cameraHandle) {};

    /**@brief Pure virtual function called only once when VK is ready to render*/
    virtual void setup() = 0;

    /**@brief Virtual function called once when VK is ready to render with camera handle
     * @param camHandle: Handle to currently connected camera
     * Called if the script type is: ArCameraScript and
     * the cameraHandle has been initialized by the CameraConnection Class */
    virtual void setup(Render r) {};

    /**@brief Pure virtual function called on every UI update, also each frame*/
    virtual void onUIUpdate(AR::GuiObjectHandles uiHandle) = 0;

    void uiUpdate(AR::GuiObjectHandles uiHandle) {

        if (renderData.finishedSetup)
            onUIUpdate(uiHandle);

    }


    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    virtual ScriptType getType() { return AR_SCRIPT_TYPE_DISABLED; }

    void drawScript(VkCommandBuffer commandBuffer, uint32_t i) {

        if (!renderData.finishedSetup)
            return;

        auto time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time_span =
                std::chrono::duration_cast<std::chrono::duration<float>>(time - lastLogTime);
        if (time_span.count() > 2.0f) {
            lastLogTime = std::chrono::high_resolution_clock::now();
            renderData.pLogger->info("Draw-count: {} | Script: {} |", renderData.scriptDrawCount, renderData.scriptName.c_str());
        }

        draw(commandBuffer, i);
        renderData.scriptDrawCount++;

    };

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i) {};

    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    void updateUniformBufferData(Render *data) {
        this->renderData.camera = data->camera;
        this->renderData.crlCamera = data->crlCamera;
        this->renderData.gui = data->gui;
        this->renderData.deltaT = data->deltaT;
        this->renderData.index = data->index;
        this->renderData.pLogger = data->pLogger;

        this->renderData.type = getType();

        renderData.scriptRuntime = (float) (std::chrono::steady_clock::now() - startTime).count();

        // Default update function is called for updating models. Else CRL extension
        if (renderData.type == AR_SCRIPT_TYPE_DEFAULT || renderData.type == AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY)
            update();
        else if (renderData.crlCamera->get()->camPtr != nullptr)
            update(renderData.crlCamera->get());
        else
            update();


        // If initialized
        if (renderUtils.uniformBuffers.empty())
            return;

        UniformBufferSet currentUB = renderUtils.uniformBuffers[renderData.index];
        if (renderData.type != AR_SCRIPT_TYPE_DISABLED) {
            // TODO unceesarry mapping and unmapping occurring here.
            currentUB.bufferOne.map();
            memcpy(currentUB.bufferOne.mapped, bufferOneData, sizeof(UBOMatrix));
            currentUB.bufferOne.unmap();

            currentUB.bufferTwo.map();
            memcpy(currentUB.bufferTwo.mapped, bufferTwoData, sizeof(FragShaderParams));
            currentUB.bufferTwo.unmap();

            if (renderData.type != AR_SCRIPT_TYPE_POINT_CLOUD) return;
            currentUB.bufferThree.map();
            memcpy(currentUB.bufferThree.mapped, bufferThreeData, sizeof(PointCloudParam));
            currentUB.bufferThree.unmap();

        }
    }

    void createUniformBuffers(RenderUtils utils, Base::Render rData) {
        if (this->getType() == AR_SCRIPT_TYPE_DISABLED)
            return;
        renderData = std::move(rData);

        renderUtils = std::move(utils);
        renderUtils.uniformBuffers.resize(renderUtils.UBCount);

        startTime = std::chrono::steady_clock::now();
        lastLogTime = std::chrono::steady_clock::now();


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


        renderData.scriptRuntime = (float) (std::chrono::steady_clock::now() - startTime).count();
        if (getType() == AR_SCRIPT_TYPE_CRL_CAMERA || getType() == AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY || getType() == AR_SCRIPT_TYPE_POINT_CLOUD)
            setup(renderData);
        else
            setup();

        renderData.finishedSetup = true;
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

protected:

private:
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::duration<float>> startTime;
    std::chrono::high_resolution_clock::time_point lastLogTime;

};

#endif //MULTISENSE_BASE_H
