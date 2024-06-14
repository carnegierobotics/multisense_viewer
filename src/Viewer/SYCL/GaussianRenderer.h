//
// Created by magnus on 5/17/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANRENDERER_H
#define MULTISENSE_VIEWER_GAUSSIANRENDERER_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <glm/ext/quaternion_float.hpp>
#include <sycl/sycl.hpp>

#include <stb_image_write.h>
#include <stb_image.h>
#include <filesystem>

#include "Viewer/Core/Camera.h"
#include "Viewer/SYCL/radixsort/RadixSorter.h"
#include "AbstractRenderer.h"
#include "Viewer/Core/RenderDefinitions.h"

namespace VkRender {


    class GaussianRenderer {
    public:

        struct GaussianPoint {
            float opacityBuffer{};
            glm::vec3 color{};
            glm::vec3 conic{};
            glm::vec3 screenPos{};
            uint32_t tileArea = 0;
            uint32_t tileInclusiveSum = 0;
            glm::vec2 bbMin, bbMax;
            float depth = 0.0f;
            float radius = 0.0f;
        };

        struct CameraParams {
            float tanFovX = 0;
            float tanFovY = 0;
            float focalX = 0;
            float focalY = 0;
        };

        struct GaussianPoints {
            std::vector<glm::vec3> positions;
            std::vector<glm::quat> quats;
            std::vector<glm::vec3> scales;
            std::vector<float> opacities;
            std::vector<float> sphericalHarmonics;  // Add this line
            uint32_t shDim = 3; // default rgb
            [[nodiscard]] uint32_t getSize() const {
                return positions.size();
            }

            uint32_t getShDim() const {
                return shDim;
            }
        };

        GaussianRenderer() = default;

        ~GaussianRenderer() {
        }

        void setup(const VkRender::AbstractRenderer::InitializeInfo &initInfo);

        void render(const AbstractRenderer::RenderInfo &info, const VkRender::RenderUtils *pUtils);

        uint8_t *getImage();

        uint32_t getImageSize();

        GaussianRenderer::GaussianPoints gs;
    public:
        static GaussianPoints loadNaive();

        static GaussianPoints loadFromFile(std::filesystem::path path, int i);

        bool loadedPly = false;

        void setupBuffers(const VkRender::Camera *camera);

    private:

        CameraParams getHtanfovxyFocal(float fovy, float h, float w) {
            float htany = std::tan(glm::radians(fovy) / 2.0f);
            float htanx = htany / h * w;
            float focal_y = h / (2.0f * htany);
            float focal_x = focal_y * (w / h); // Ensure aspect ratio is maintained

            return {htanx, htany, focal_x, focal_y};
        }

    private:
        VkRender::AbstractRenderer::InitializeInfo m_initInfo;
        uint8_t *m_image{};

        sycl::queue queue{};
        sycl::buffer<glm::vec3, 1> positionBuffer{0};
        sycl::buffer<glm::vec3, 1> scalesBuffer{0};
        sycl::buffer<glm::quat, 1> quaternionBuffer{0};
        sycl::buffer<float, 1> opacityBuffer{0};
        sycl::buffer<float, 1> sphericalHarmonicsBuffer{0};

        sycl::buffer<glm::vec3, 1> covariance2DBuffer{0};
        sycl::buffer<glm::vec3, 1> conicBuffer{0};
        sycl::buffer<glm::vec3, 1> screenPosBuffer{0};
        sycl::buffer<glm::mat3, 1> covarianceBuffer{0};
        sycl::buffer<glm::vec3, 1> colorOutputBuffer{0};

        sycl::buffer<uint32_t, 1> keysBuffer{0};
        sycl::buffer<uint32_t, 1> valuesBuffer{0};

        sycl::event finishedRenderEvent;


        sycl::buffer<GaussianPoint, 1> pointsBuffer{0};
        sycl::buffer<uint32_t, 1> numTilesTouchedBuffer{0};
        sycl::buffer<uint32_t, 1> numTilesTouchedInclusiveSumBuffer{0};
        // Optimization buffers
        sycl::buffer<bool, 1> activeGSBuffer{0};

        std::unique_ptr<crl::RadixSorter> radixSorter;


        sycl::buffer<uint8_t, 3> pngImageBuffer{sycl::range<3>()};
        sycl::buffer<uint8_t, 3> imageBuffer{sycl::range<3>()};
        sycl::buffer<uint8_t, 3> imageBuffer2{sycl::range<3>()};
        uint32_t width{}, height{};
        std::vector<uint8_t> flattenedImage;
    };

}

#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
