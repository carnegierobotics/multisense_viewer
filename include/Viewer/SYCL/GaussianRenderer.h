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

#include <stb_image_write.h>
#include <stb_image.h>

#include "Viewer/Core/Camera.h"

class GaussianRenderer {
public:

    struct GaussianPoints{
        std::vector<glm::vec3> positions;
        std::vector<glm::quat> quats;
        std::vector<glm::vec3> scales;
        std::vector<float> opacities;

        [[nodiscard]] uint32_t getSize() const{
            return positions.size();
        }
    };

    explicit GaussianRenderer(const VkRender::Camera &camera);

private:

    GaussianPoints loadNaive();

    void simpleRasterizer(const VkRender::Camera &camera);

    std::vector<float> getHtanfovxyFocal(float fovy, float h, float w) {
        float htany = std::tan(glm::radians(fovy) / 2.0f);
        float htanx = htany / h * w;
        float focal_y = h / (2.0f * htany);
        float focal_x = focal_y * (w / h); // Ensure aspect ratio is maintained
        return {htanx, htany, focal_x, focal_y};
    }

};


#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
