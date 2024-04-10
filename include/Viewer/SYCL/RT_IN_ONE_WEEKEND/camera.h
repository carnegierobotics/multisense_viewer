//
// Created by magnus on 4/3/24.
//

#ifndef SYCL_APP_CAMERA_H
#define SYCL_APP_CAMERA_H

#include "vec3.h"


struct Camera {
    vec3 center;
    vec3 pixel00Loc;
    vec3 upperLeft;
    vec3 pixelDeltaU;
    vec3 pixelDeltaV;


};
#endif //SYCL_APP_CAMERA_H
