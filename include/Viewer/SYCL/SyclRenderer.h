//
// Created by magnus on 4/10/24.
//

#ifndef MULTISENSE_VIEWER_SYCLRENDERER_H
#define MULTISENSE_VIEWER_SYCLRENDERER_H

#include <sycl/sycl.hpp>
#include "Viewer/SYCL/RT_IN_ONE_WEEKEND/vec3.h"
#include "Viewer/SYCL/RT_IN_ONE_WEEKEND/camera.h"

class SyclRenderer {

public:

    SyclRenderer();


    void save_image(vec3 *fb_data, const std::string &filename, uint32_t width, uint32_t height);

    // allocate the frame buffer on the Host
    std::vector<vec3> fb; // frame buffer
};


#endif //MULTISENSE_VIEWER_SYCLRENDERER_H
