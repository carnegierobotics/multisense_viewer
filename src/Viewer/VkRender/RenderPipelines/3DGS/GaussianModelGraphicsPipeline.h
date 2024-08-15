//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H


#include "Viewer/VkRender/Core/CommandBuffer.h"

#ifdef SYCL_ENABLED
#include <sycl/sycl.hpp>
#include "Viewer/VkRender/RenderPipelines/3DGS/radixsort/RadixSorter.h"
#include "Viewer/SYCL/RasterizerUtils.h"
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/VkRender/Core/Texture.h"

#endif

namespace VkRender {
#ifdef SYCL_ENABLED
    class GaussianModelGraphicsPipeline {
    public:
        explicit GaussianModelGraphicsPipeline(VulkanDevice &vulkanDevice);

        ~GaussianModelGraphicsPipeline();

        template<typename T>
        void bind(T &modelComponent, Camera& camera);
        void generateImage(Camera &camera);

        void draw(CommandBuffer &cmdBuffers);

        std::shared_ptr<TextureVideo> getTextureRenderTarget() {return m_textureVideo;}

        uint8_t *getImage();
        uint32_t getImageSize();
    private:

        VulkanDevice& m_vulkanDevice;
        sycl::queue queue{};
        bool m_boundBuffers = false;
        uint8_t *m_image = nullptr;
        uint32_t m_imageSize = 0;

        glm::vec3 *positionBuffer = nullptr;
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
        uint32_t width{}, height{};

        uint32_t m_numPoints = 0;
        uint32_t m_shDim = 0;

        void logTimes(std::chrono::duration<double, std::milli> t1, std::chrono::duration<double, std::milli> t2,
                      std::chrono::duration<double, std::milli> t3, std::chrono::duration<double, std::milli> t4,
                      std::chrono::duration<double, std::milli> t5, std::chrono::duration<double, std::milli> t6,
                      std::chrono::duration<double, std::milli> t7, std::chrono::duration<double, std::milli> t8,
                      std::chrono::duration<double, std::milli> t9, bool error);


        std::shared_ptr<TextureVideo> m_textureVideo;
    };
#else
    class GaussianModelGraphicsPipeline {
    public:
        void draw(CommandBuffer&) { /* no-op */ }
        template <typename T>
        void bind(T&) { /* no-op */ }
    };
#endif
}

#endif //MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H
