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

    struct GaussianPoints {
        std::vector<glm::vec3> positions;
        std::vector<glm::quat> quats;
        std::vector<glm::vec3> scales;
        std::vector<float> opacities;

        [[nodiscard]] uint32_t getSize() const {
            return positions.size();
        }
    };

    explicit GaussianRenderer(const VkRender::Camera &camera);
    ~GaussianRenderer(){
        if (image)
            stbi_image_free(image);
    }

    void simpleRasterizer(const VkRender::Camera &camera);

    uint8_t *img{};

private:

    static GaussianPoints loadNaive();


    std::vector<float> getHtanfovxyFocal(float fovy, float h, float w) {
        float htany = std::tan(glm::radians(fovy) / 2.0f);
        float htanx = htany / h * w;
        float focal_y = h / (2.0f * htany);
        float focal_x = focal_y * (w / h); // Ensure aspect ratio is maintained
        return {htanx, htany, focal_x, focal_y};
    }

private:
    uint8_t *image;

    sycl::queue queue{};
    GaussianRenderer::GaussianPoints gs;
    sycl::buffer<glm::vec3, 1> positionBuffer{0};
    sycl::buffer<glm::vec3, 1> scalesBuffer{0};
    sycl::buffer<glm::quat, 1> quaternionBuffer{0};
    sycl::buffer<float, 1> opacityBuffer{0};

    sycl::buffer<glm::vec3, 1> covariance2DBuffer{0};
    sycl::buffer<glm::vec3, 1> conicBuffer{0};
    sycl::buffer<glm::vec3, 1> screenPosBuffer{0};
    sycl::buffer<glm::mat3, 1> covarianceBuffer{0};

    sycl::buffer<uint8_t, 3> pngImageBuffer{sycl::range<3>()};
    sycl::buffer<uint8_t, 3> imageBuffer{sycl::range<3>()};
};


#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
