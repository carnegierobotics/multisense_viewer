//
// Created by magnus on 4/10/24.
//

#ifndef MULTISENSE_VIEWER_SYCLRAYTRACER_H
#define MULTISENSE_VIEWER_SYCLRAYTRACER_H


#include <sycl/sycl.hpp>
#include "Viewer/SYCL/RT_IN_ONE_WEEKEND/vec3.h"
#include "Viewer/SYCL/RT_IN_ONE_WEEKEND/camera.h"
#include "Viewer/SYCL/RT_IN_ONE_WEEKEND/sphere.h"

class SYCLRayTracer {

public:
    struct Pixel {
        uint8_t r, g, b, a = 1;
    };
    SYCLRayTracer(int width, int height);

    void save_image(const std::string &filename, uint32_t width, uint32_t height);
    // allocate the frame buffer on the Host
    std::vector<vec3> fb; // frame buffer
    void render(int width, int height, int num_spheres, sycl::queue &queue, vec3 *fb_data, const sphere *spheres,
                Camera *camera);

    std::vector<Pixel> get_image_8bit(uint32_t width, uint32_t height);
};


#endif //MULTISENSE_VIEWER_SYCLRAYTRACER_H
