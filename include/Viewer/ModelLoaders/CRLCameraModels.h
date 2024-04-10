/**
 * @file: MultiSense-Viewer/include/Viewer/ModelLoaders/CRLCameraModels.h
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
 *   2022-3-10, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_CRLCAMERAMODELS_H
#define MULTISENSE_CRLCAMERAMODELS_H


#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <MultiSense/MultiSenseTypes.hh>

#include "Viewer/Scripts/Private/TextureDataDef.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Tools/Macros.h"

/***
 * @brief Bridges the gap between rendering vulkan images with data from MultiSense cameras. Tightly integrated with \refitem VkRender::TextureData
 */
class CRLCameraModels {

public:
    CRLCameraModels() = default;

    ~CRLCameraModels()  = default;
    struct Model {
        explicit Model(const VkRender::RenderUtils *renderUtils, uint32_t framesInFlight);

        ~Model();

        std::vector<VkDescriptorSet> m_Descriptors;
        std::vector<VkDescriptorSetLayout> m_DescriptorSetLayout{};
        VkDescriptorPool m_DescriptorPool{};
        std::vector<VkPipeline> m_Pipeline{};
        std::vector<VkPipeline> m_SelectionPipeline{};
        bool m_InitializedPipeline = false;

        std::vector<VkPipelineLayout> m_PipelineLayout{};
        std::vector<VkPipelineLayout> m_SelectionPipelineLayout{};

        /**@brief Property to flashing/disable drawing of this m_Model. Set to false if you want to control when to draw the m_Model. */
        bool m_Draw = true;
        VkRender::CRLCameraDataType m_CameraDataType{};

        struct Mesh {
            VulkanDevice *device = nullptr;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;
            struct Vertices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            };
            struct Indices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            };
            std::vector<Vertices> vertices;
            std::vector<Indices> indices;
            Buffer uniformBuffer{};
        } m_Mesh{};

        VulkanDevice *m_VulkanDevice{};
        uint32_t m_FramesInFlight = 1;

        std::vector<std::unique_ptr<TextureVideo>> m_TextureVideo;       // Default texture that gets updated usually on each camera frame update
        std::vector<std::unique_ptr<TextureVideo>> m_PointCloudTexture;  // pc color texture
        std::vector<std::unique_ptr<TextureVideo>> m_TextureChromaU;     // supporting tex if no ycbcr sampler present (old hw)
        std::vector<std::unique_ptr<TextureVideo>> m_TextureChromaV;     // supporting tex if no ycbcr sampler is present (oldhw)
        std::vector<Buffer> m_ColorPointCloudBuffer;
        std::vector<Texture2D> *m_TextureComputeTarget;
        std::vector<Texture3D> *m_TextureComputeTarget3D;

        void
        createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                              const std::vector<uint32_t> &indices = std::vector<uint32_t>());

        void createEmptyTexture(uint32_t width, uint32_t height, VkRender::CRLCameraDataType texType, bool forPointCloud = false, int i = 0);

        bool updateTexture(VkRender::CRLCameraDataType type, uint32_t currentFrame);

        bool getTextureDataPointers(VkRender::TextureData *tex, uint32_t currentFrame) const;

        bool updateTexture(VkRender::TextureData *tex, uint32_t currentFrame);
    };

    /**
     * Call to draw m_Model
     * @param commandBuffer handle to commandbuffer to record drawing command
     * @param i index of swapchain m_Image to render to
     * @param model modeol to draw
     * @param b if we want to render additional pipeline
     */
    void draw(CommandBuffer * commandBuffer, uint32_t i, Model *model, bool b = true);

protected:

    const VulkanDevice *vulkanDevice = nullptr;

    uint32_t m_SwapChainImageCount = 0;
    std::vector<VkPipelineShaderStageCreateInfo> m_Shaders;

    /***
     * @brief Helper function
     * @param pModel
     */
    void createDescriptorSetLayout(Model *pModel);


    /**
     * @brief Helper function
     * Create the pipeline layout
     * @param pT pointer to store pipelinelayout object
     */
    void createPipelineLayout(VkPipelineLayout *pT, VkDescriptorSetLayout *layout, uint32_t numDescriptorLayouts);

    /**
     * @brief Bind a default m_Descriptor layout to the pipeline for images
     * @param model Which m_Model to configure
     * @param ubo reference to uniform buffers
     */
    void createImageDescriptors(Model *model, const std::vector<VkRender::UniformBufferSet> &ubo);

    /**
     * Bind a default m_Descriptor layout fo the pipline for point clouds
     * @param model Which m_Model to configure
     * @param ubo reference to uniform buffers
     */
    void createPointCloudDescriptors(Model *model, const std::vector<VkRender::UniformBufferSet> &ubo);

    /**
     * Create render pipeline
     * @param pT Handle to render pass
     * @param vector vector with shaders
     * @param type what kind of camera data we expect to handle
     * @param pPipelineT additional pipeline
     * @param pLayoutT additional pipeline layout
     */
    void createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, VkRender::CRLCameraDataType type,
                        VkPipeline *pPipelineT, VkPipelineLayout *pLayoutT, VkSampleCountFlagBits bits);

    /**
     * Create descriptors for this m_Model
     * @param count number of descriptorsets needed
     * @param ubo reference to uniform buffers
     * @param model which m_Model to configure
     */
    void createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo, Model *model);

    /**
     * Creates the render pipeline using helper functions from this class
     * @param vector vector of shaders
     * @param model m_Model to configure
     * @param renderUtils handle to render utilities from the scripts base \refitem Base class
     */
    void
    createRenderPipeline(const std::vector<VkPipelineShaderStageCreateInfo> &vector, Model *model,
                         const VkRender::RenderUtils *renderUtils);

};


#endif //MULTISENSE_CRLCAMERAMODELS_H
