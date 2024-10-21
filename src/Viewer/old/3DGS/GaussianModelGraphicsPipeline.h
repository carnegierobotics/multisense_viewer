//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H


#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"
#include "Viewer/VkRender/Core/Camera.h"
#ifdef SYCL_ENABLED
#include <sycl/sycl.hpp>
#include "Viewer/VkRender/RenderResources/3DGS/radixsort/RadixSorter.h"
#include "Viewer/VkRender/RenderResources/3DGS/RasterizerUtils.h"
#endif

namespace VkRender {
#ifdef SYCL_ENABLED
    class GaussianModelGraphicsPipeline {
    public:
        explicit GaussianModelGraphicsPipeline(VulkanDevice &vulkanDevice, RenderPassInfo& renderPassInfo,  uint32_t width, uint32_t height);

        ~GaussianModelGraphicsPipeline();

        void generateImage(Camera &camera, int b);

        void draw(CommandBuffer &cmdBuffers);

        std::shared_ptr<TextureVideo> getTextureRenderTarget() {return m_textureVideo;}

        uint8_t *getImage();
        uint32_t getImageSize();


        void update(uint32_t currentFrame);

        void updateTransform(TransformComponent &transform);

        void updateView(const Camera &camera);

        void bind(GaussianModelComponent &modelComponent);
        void bind(MeshComponent *meshComponent);
        void setTexture(const VkDescriptorImageInfo *info);

    private:
        sycl::queue queue{};
        bool m_boundBuffers = false;
        uint8_t *m_image = nullptr;
        uint32_t m_imageSize = 0;

        glm::vec3 *positionBuffer = nullptr;
        glm::vec3 *normalsBuffer = nullptr;
        glm::vec3 *scalesBuffer = nullptr;
        glm::quat *quaternionBuffer = nullptr;
        float *opacityBuffer = nullptr;
        float *sphericalHarmonicsBuffer = nullptr;

        Rasterizer::GaussianPoint *pointsBuffer = nullptr;
        uint32_t *numTilesTouchedBuffer = nullptr;
        uint32_t *pointOffsets = nullptr;
        uint32_t *keysBuffer = nullptr;
        uint32_t *valuesBuffer = nullptr;

        glm::ivec2 *rangesBuffer = nullptr;
        uint8_t *imageBuffer = nullptr;

        std::unique_ptr<Sorter> sorter;
        uint32_t m_width{}, m_height{};

        uint32_t m_numPoints = 0;
        uint32_t m_shDim = 0;

        void logTimes(std::chrono::duration<double, std::milli> t1, std::chrono::duration<double, std::milli> t2,
                      std::chrono::duration<double, std::milli> t3, std::chrono::duration<double, std::milli> t4,
                      std::chrono::duration<double, std::milli> t5, std::chrono::duration<double, std::milli> t6,
                      std::chrono::duration<double, std::milli> t7, std::chrono::duration<double, std::milli> t8,
                      std::chrono::duration<double, std::milli> t9, bool error);


        std::shared_ptr<TextureVideo> m_textureVideo;


        //
        struct Vertices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t vertexCount = 0;
        };
        struct Indices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t indexCount = 0;
        };

        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        Texture2D m_emptyTexture;
        Texture2D m_objTexture;

        Indices indices{};
        Vertices vertices{};

        std::string m_vertexShader;
        std::string m_fragmentShader;

        UBOMatrix m_vertexParams; // Non GPU-accessible data, shared across frames
        FragShaderParams m_fragParams; // Non GPU-accessible data, shared across frames
        std::vector<DefaultRenderData> m_renderData;
        SharedRenderData m_sharedRenderData;

        void setupUniformBuffers();

        void setupDescriptors();

        void setupPipeline();

        void cleanUp();

    };
#else
    class GaussianModelGraphicsPipeline {
    public:

        GaussianModelGraphicsPipeline(VulkanDevice &vulkanDevice, RenderPassInfo& renderPassInfo,  uint32_t width, uint32_t height) {}
        ~GaussianModelGraphicsPipeline() = default;

        void generateImage(Camera &camera, int i) {
        }


        void draw(CommandBuffer &cmdBuffers) {

        }

        std::shared_ptr<TextureVideo> getTextureRenderTarget() {return nullptr;}

        uint8_t *getImage() {
            return nullptr;
        }
        uint32_t getImageSize() {
            return 0;
        }


        void update(uint32_t currentFrame) {

        }

        void updateTransform(TransformComponent &transform) {

        }

        void updateView(const Camera &camera) {

        }

        void bind(GaussianModelComponent &modelComponent) {

        }
        void bind(MeshComponent *meshComponent) {

        }
        void setTexture(const VkDescriptorImageInfo *info) {

        }

    };
#endif
}

#endif //MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H
