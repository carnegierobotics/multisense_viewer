//
// Created by magnus on 11/27/21.
//

#ifndef MULTISENSE_BASE_H
#define MULTISENSE_BASE_H

#include <filesystem>
#include <utility>
#include "Camera.h"
#include "MultiSense/src/tools/Utils.h"
#include "MultiSense/src/tools/Logger.h"
#include "GLFW/glfw3.h"

#define INTERVAL_10_SECONDS_LOG_DRAW_COUNT 10
#define NUM_POINTS 2048 // Changing this also needs to be changed in the vs shader.


class CameraConnection; // forward declaration of this class to speed up compile time. Separate scripts/model_loaders from ImGui source recompile

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


class Base {
public:

    /**@brief A standard set of uniform buffers */
    struct UniformBufferSet {
        Buffer bufferOne;
        Buffer bufferTwo;
        Buffer bufferThree;
        Buffer bufferFour;
    };

    // TODO DELETE POINTERS AS WELL
    void *bufferOneData{};
    void *bufferTwoData{};
    void *bufferThreeData{};
    void *bufferFourData{};

    struct RenderUtils {
        VulkanDevice *device{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};

        // TODO Some error happening on destruction on these vectors. Identify or made change to use pointers and free the later
        std::vector<VkPipelineShaderStageCreateInfo> shaders;
        std::vector<UniformBufferSet> uniformBuffers;


    } renderUtils;

    struct Render {
        uint32_t index;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        std::unique_ptr<CameraConnection> *crlCamera = nullptr;
        ScriptType type;
        const std::vector<AR::Element> *gui;
        Log::Logger *pLogger;

        uint32_t height;
        uint32_t width;
        const int *lastKeyPress;
        const int* lastKeyAction;
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

    virtual void onWindowResize(AR::GuiObjectHandles uiHandle) {

    };

    void windowResize(Render *data, AR::GuiObjectHandles uiHandle) {
        updateRenderData(data);

        onWindowResize(uiHandle);
    }

    /**@brief Pure virtual function called on every UI update, also each frame*/
    virtual void onUIUpdate(AR::GuiObjectHandles uiHandle) = 0;

    void uiUpdate(AR::GuiObjectHandles uiHandle) {
        auto p = this->renderData;

        if (!this->renderData.drawThisScript)
            return;

        for (auto &dev: *uiHandle.devices) {
            if (dev.state == AR_STATE_ACTIVE) {
                onUIUpdate(uiHandle);
            }
        }
    }


    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    virtual ScriptType getType() { return AR_SCRIPT_TYPE_DISABLED; }

    void drawScript(VkCommandBuffer commandBuffer, uint32_t i) {

        if (!renderData.drawThisScript)
            return;

        auto time = std::chrono::steady_clock::now();
        std::chrono::duration<float> time_span =
                std::chrono::duration_cast<std::chrono::duration<float>>(time - lastLogTime);
        if (time_span.count() > INTERVAL_10_SECONDS_LOG_DRAW_COUNT) {
            lastLogTime = std::chrono::steady_clock::now();
            renderData.pLogger->info("Draw-number: {} for script: {}", (int) renderData.scriptDrawCount,
                                     renderData.scriptName.c_str());
        }


        draw(commandBuffer, i);

        if (i == 0)
            renderData.scriptDrawCount++;

    };

    virtual void draw(VkCommandBuffer commandBuffer, uint32_t i) {};

    /**@brief Which script type this is. Can be used to enable/disable rendering of this script */
    void updateUniformBufferData(Render *data) {
        updateRenderData(data);

        renderData.scriptRuntime = (float) (std::chrono::steady_clock::now() - startTime).count();



        // Default update function is called for updating models. Else CRL extension
        if (renderData.type == AR_SCRIPT_TYPE_DEFAULT || renderData.type == AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY)
            update();
        else
            update(renderData.crlCamera->get());



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

            currentUB.bufferThree.map();
            memcpy(currentUB.bufferThree.mapped, bufferThreeData, sizeof(PointCloudParam));
            currentUB.bufferThree.unmap();

            currentUB.bufferFour.map();
            memcpy(currentUB.bufferFour.mapped, bufferFourData, sizeof(float));
            currentUB.bufferFour.unmap();


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


        bufferOneData = new UBOMatrix();
        bufferTwoData = new FragShaderParams();
        bufferThreeData = new PointCloudParam();
        bufferFourData = new PointCloudParam();

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

            renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                             &uniformBuffer.bufferFour, sizeof(float));

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
        onDestroy();
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
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime;
    std::chrono::steady_clock::time_point lastLogTime;


    void updateRenderData(Render *data) {
        this->renderData.camera = data->camera;
        this->renderData.crlCamera = data->crlCamera;
        this->renderData.gui = data->gui;
        this->renderData.deltaT = data->deltaT;
        this->renderData.index = data->index;
        this->renderData.pLogger = data->pLogger;
        this->renderData.height = data->height;
        this->renderData.width = data->width;
        this->renderData.type = getType();
        this->input.lastKeyPress = data->lastKeyPress;
        this->input.action = data->lastKeyAction;

    }


    struct Input {

        [[nodiscard]] bool getButtonDown(int key) const {
            if (key == *lastKeyPress && *action == GLFW_PRESS)
                return true;

            return false;
        }

        [[nodiscard]] bool getButtonUp(int key) const {
            if (key == *lastKeyPress && *action == GLFW_RELEASE)
                return true;

            return false;
        }

        [[nodiscard]] bool getButton(int key) const {
            if (key == *lastKeyPress)
                return true;

            return false;
        }

        const int* action;
        const int* lastKeyPress;

    } input;
};

#endif //MULTISENSE_BASE_H
