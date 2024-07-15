/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Core/Definitions.h
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

#include "Viewer/VkRender/Core/Buffer.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Core/KeyInput.h"

// Predeclare to speed up compile times
namespace VkRender {
    class Camera;

}

namespace VkRender {


    /**
     * @brief GLFW and Vulkan combination to create a SwapChain
     */
    typedef struct SwapChainCreateInfo {
        GLFWwindow *pWindow{};
        bool vsync = true;
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

        bool operator==(const Vertex& other) const {
            return pos == other.pos && color == other.color && uv0 == other.uv0;
        }
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
        glm::vec3 camPos{};
    };

    struct UBOCamera {
        std::array<glm::vec4, 21> positions;
    };

    struct ShaderValuesParams {
        glm::vec4 lightDir{};
        float exposure = 4.5f;
        float gamma = 2.2f;
        float prefilteredCubeMipLevels;
        float scaleIBLAmbient = 1.0f;
        float debugViewInputs = 0;
        float debugViewEquation = 0;
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

    /** @brief RenderData which are shared across render passes */
    struct DefaultRenderData {
        Buffer fragShaderParamsBuffer;                 // GPU Accessible
        Buffer mvpBuffer;                              // GPU Accessible

        std::vector<VkDescriptorSet> descriptorSets;
        VkDescriptorPool descriptorPool{};
        VkDescriptorSetLayout descriptorSetLayout{};

        std::unordered_map<RenderPassType, VkPipeline> pipeline{};
        std::unordered_map<RenderPassType, VkPipelineLayout> pipelineLayout{};

        std::unordered_map<RenderPassType, bool> busy{};
        std::unordered_map<RenderPassType, bool> requestIdle{};


        bool isBusy() const {
            return std::any_of(busy.begin(), busy.end(), [](const auto& item) { return item.second; });
        }

    };

    struct EditorRenderPass {
        std::vector<VkFramebuffer> frameBuffers{};
        VkRenderPass renderPass = VK_NULL_HANDLE;
        Camera *camera{};
        struct {
            VkImage image = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
        } depthStencil{};
        struct {
            VkImage image = VK_NULL_HANDLE;
            VkImage resolvedImage = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            VkDeviceMemory resolvedMem = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkImageView resolvedView = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
        } colorImage{};

        bool idle = true;
        VkDescriptorImageInfo imageInfo{};
        VkDescriptorImageInfo depthImageInfo{};
        std::string type;
        bool multisampled = true;
        bool setupFrameBuffer = true;
        std::string debugName = "EditorRenderPass";
    };

    /** Containing Basic Vulkan Resources for rendering for use in scripts **/
    struct RenderUtils {
        VulkanDevice *device{};
        VkInstance *instance{};
        VkRenderPass *renderPass{};
        Input* input{};
        uint32_t height = 0;
        uint32_t width = 0;

        // Multiple viewpoint (Off screen rendering)
        VkFormat swapchainColorFormat {};
        VkFormat depthFormat {};
        uint32_t swapchainIndex = 0;
        VkQueue graphicsQueue;
        VkSampleCountFlagBits msaaSamples;
        uint32_t swapchainImages = 0; // TODO rename to swapchain images

    };


}


#endif //MULTISENSE_DEFINITIONS_H
