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

#include "Viewer/Core/Camera.h"

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

    // Define Splat struct
    struct Splat {
        float depth;
        int tileIndex;

        bool operator<(const Splat& other) const {
            return depth < other.depth;
        }
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

    explicit GaussianRenderer(const VkRender::Camera &camera);
    ~GaussianRenderer(){
        if (image)
            stbi_image_free(image);
    }
    void tileRasterizer(const VkRender::Camera &camera);

    uint8_t *img{};

    static GaussianPoints loadNaive();
    static GaussianPoints loadFromFile(std::filesystem::path path, int i);
    GaussianRenderer::GaussianPoints gs;
    bool loadedPly = false;
    void setupBuffers(const VkRender::Camera &camera);

private:

    CameraParams getHtanfovxyFocal(float fovy, float h, float w) {
        float htany = std::tan(glm::radians(fovy) / 2.0f);
        float htanx = htany / h * w;
        float focal_y = h / (2.0f * htany);
        float focal_x = focal_y * (w / h); // Ensure aspect ratio is maintained

        return {htanx, htany, focal_x, focal_y};
    }

private:
    uint8_t *image;

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

    // Optimization buffers
    sycl::buffer<bool, 1> activeGSBuffer{0};


    sycl::buffer<uint8_t, 3> pngImageBuffer{sycl::range<3>()};
    sycl::buffer<uint8_t, 3> imageBuffer{sycl::range<3>()};
    uint32_t width, height;
    std::vector<uint8_t> flattenedImage;
};


#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
