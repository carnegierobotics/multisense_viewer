/**
 * @file: MultiSense-Viewer/include/Viewer/Core/Definitions.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2021-05-15, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
    #include <windows.h>
#endif

#include <unordered_map>
#include <memory>
#include <utility>
#include <array>
#include <vulkan/vulkan_core.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#ifdef APIENTRY
#undef APIENTRY
#endif
#include <GLFW/glfw3.h>

#include "Viewer/Core/Buffer.h"
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"

#define INTERVAL_10_SECONDS 10
#define INTERVAL_1_SECOND 1
#define INTERVAL_2_SECONDS 2
#define INTERVAL_5_SECONDS 5
#define MAX_IMAGES_IN_QUEUE 5


typedef uint32_t VkRenderFlags;


// Predeclare to speed up compile times
namespace VkRender {
    class Camera;
    namespace MultiSense {
        class CRLPhysicalCamera;
    }
}

namespace VkRender {

    /**
 * @brief Defines draw behaviour of a script
 */
    typedef enum ScriptType {
        /** Do not draw script at all */
        CRL_SCRIPT_TYPE_DISABLED = 0x00,
        /** Draw script since first frame and allocate additional MVP buffers */ //TODO Delete if possible
        CRL_SCRIPT_TYPE_ADDITIONAL_BUFFERS = 0x01,
        /** Draw script after crl camera connect */
        CRL_SCRIPT_TYPE_DEFAULT = 0x02,
        /** Draw script since application startup in the Renderer3D. No particular order */
        CRL_SCRIPT_TYPE_RENDERER3D = 0x04,
        /** Create this script before default and always render this type first. No internal ordering amongst scripts */
        CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE = 0x08,
        /** Draw script since application startup in the Simulated camera. No particular order */
        CRL_SCRIPT_TYPE_SIMULATED_CAMERA = 0x10,

    } ScriptType;
    typedef VkRenderFlags ScriptTypeFlags;

    typedef enum DrawMethod {

        CRL_SCRIPT_DONT_DRAW,
        CRL_SCRIPT_DRAW,
        /** CRL_SCRIPT_TYPE_RELOAD This script is set to reload (run onDestroy and Create funcs) next frame after this is set*/
        CRL_SCRIPT_RELOAD
    } DrawMethod;

/**
 * @brief Labels data coming from the camera to a type used to initialize textures with various formats and samplers
 */
    typedef enum CRLCameraDataType {
        CRL_DATA_NONE,
        CRL_GRAYSCALE_IMAGE,
        CRL_COLOR_IMAGE_RGBA,
        CRL_COLOR_IMAGE_YUV420,
        CRL_CAMERA_IMAGE_NONE,
        CRL_DISPARITY_IMAGE,
        CRL_POINT_CLOUD,
        CRL_COMPUTE_SHADER,
    } CRLCameraDataType;


    /**
     * @brief GLFW and Vulkan combination to create a SwapChain
     */
    typedef struct SwapChainCreateInfo {
        GLFWwindow *pWindow{};
        bool vsync = false;
        VkInstance instance{};
        VkPhysicalDevice physicalDevice{};
        VkDevice device{};
    } SwapChainCreateInfo;

    /**
     * @brief Default Vertex information
     */
    struct Vertex {
        glm::vec3 pos{};
        glm::vec3 normal{};
        glm::vec2 uv0{};
        glm::vec2 uv1{};
        glm::vec4 joint0{};
        glm::vec4 weight0{};
        glm::vec4 color{};
    };

    /**
     * @brief MouseButtons user input
     */
    struct MouseButtons {
        bool left = false;
        bool right = false;
        bool middle = false;
        int action = 0;
        float wheel = 0.0f; // to initialize arcball zoom
        float dx = 0.0f;
        float dy = 0.0f;
        struct {
            float x = 0.0f;
            float y = 0.0f;
        } pos;
    };

    /**
     * @brief Default MVP matrices
     */
    struct UBOMatrix {
        glm::mat4 projection{};
        glm::mat4 view{};
        glm::mat4 model{};
        glm::vec3 camPos;
    };

    struct Particle {
        glm::vec2 position;
        glm::vec2 velocity;
        glm::vec4 color;
    };

    /**
     * @brief Basic lighting params for simple light calculation
     */
    struct FragShaderParams {
        glm::vec4 lightDir{};
        glm::vec4 zoomCenter{};
        glm::vec4 zoomTranslate{};
        float exposure = 4.5f;
        float gamma = 2.2f;
        float prefilteredCubeMipLevels = 0.0f;
        float scaleIBLAmbient = 1.0f;
        float debugViewInputs = 0.0f;
        float lod = 0.0f;
        glm::vec2 pad{};
        glm::vec4 disparityNormalizer; // (0: should normalize?, 1: min value, 2: max value, 3 pad)
        glm::vec4 kernelFilters; // 0 Sobel/Edge kernel, Blur kernel,
        float dt = 0.0f;
    };

    struct SkyboxTextures {
        std::shared_ptr<TextureCubeMap> environmentMap{};
        std::shared_ptr<TextureCubeMap> irradianceCube{};
        std::shared_ptr<TextureCubeMap> prefilterEnv{};
        std::shared_ptr<Texture2D> lutBrdf{};
        float prefilteredCubeMipLevels = 0;
    };

    /**
     * @brief Memory block for point clouds
     */
    struct PointCloudParam {
        /** @brief Q Matrix. See PointCloudUtility in LibMultiSense for calculation of this */
        glm::mat4 Q{};
        /** @brief Width of depth image*/
        float width{};
        /** @brief Height of depth image*/
        float height{};
        /** @brief Max disparity of image*/
        float disparity{};
        /** @brief Distance between left and right camera (tx)*/
        float focalLength{};
        /** @brief Scaling factor used when operating in cropped mode assuming uniform scaling in x- and y direction */
        float scale{};
        /** @brief Point size to view the point cloud. Larger for more distant points and smaller for closer points */
        float pointSize{};
    };

    struct ColorPointCloudParams {
        glm::mat4 instrinsics{};
        glm::mat4 extrinsics{};
        float useColor = true;
        float hasSampler = false;
    };

    /** @brief Additional default buffers for rendering mulitple models with distrinct MVP */
    struct RenderDescriptorBuffers {
        UBOMatrix mvp{};
        FragShaderParams light{};
    };

    /**@brief A standard set of uniform buffers. All current shaders can get away with using a combination of these two */
    struct RenderDescriptorBuffersData {
        Buffer mvp{};
        Buffer light{};
    };
    /**
     * @brief TODO: Unfinished. Put mouse cursor position information into shader. Meant for zoom feature
     */
    struct MousePositionPushConstant {
        glm::vec2 position{};
    };


    /**
     * @brief Vulkan resources for the cursor information pipeline
     */
    struct ObjectPicking {
        // Global render pass for frame buffer writes
        VkRenderPass renderPass{};
        // List of available frame buffers (same as number of swap chain images)
        VkFramebuffer frameBuffer{};
        VkImage colorImage{};
        VkImage depthImage{};
        VkImageView colorView{};
        VkImageView depthView{};
        VkDeviceMemory colorMem{};
        VkDeviceMemory depthMem{};
    };


    /**@brief A standard set of uniform buffers. Each script is initialized with these */
    struct UniformBufferSet {
        Buffer bufferOne{};
        Buffer bufferTwo{};
        Buffer bufferThree{};
    };

    /** Containing Basic Vulkan Resources for rendering for use in scripts **/
    struct RenderUtils {
        VulkanDevice *device{};
        VkInstance* instance{};
        uint8_t* vkDeviceUUID = nullptr; // array with size VK_UUID_SIZE

        uint32_t UBCount = 0; // TODO rename to swapchain iamges
        VkRenderPass *renderPass{};
        VkSampleCountFlagBits msaaSamples;
        std::vector<UniformBufferSet> uniformBuffers{};
        const VkRender::ObjectPicking *picking = nullptr;
        struct {
            std::shared_ptr<TextureCubeMap> irradianceCube = nullptr;
            std::shared_ptr<TextureCubeMap> prefilterEnv = nullptr;
            std::shared_ptr<Texture2D> lutBrdf = nullptr;
            float prefilteredCubeMipLevels = 0.0f;
        } skybox;
        std::mutex *queueSubmitMutex;
        const std::vector<VkFence>* fence;
    };

    /**@brief grouping containing useful pointers used to render scripts. This will probably change frequently as the viewer grows **/
    struct RenderData {

        uint32_t index = 0;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        /**
         * @brief Runtime measured in seconds
         */
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        MultiSense::CRLPhysicalCamera *crlCamera{};
        ScriptTypeFlags type{};
        uint32_t height = 0;
        uint32_t width = 0;
        bool additionalBuffers = false;
        void* streamToRun;

    };

}


#endif //MULTISENSE_DEFINITIONS_H
