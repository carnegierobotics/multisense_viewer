//
// Created by magnus on 4/3/24.
//

#ifndef SYCL_APP_KERNELS_H
#define SYCL_APP_KERNELS_H

#include <sycl/sycl.hpp>
#include "sphere.h"
#include "ray.h"
#include "hittable.h"
#include "camera.h"

using real_t = float;

class simple_rng {
public:
    using result_type = uint64_t;

    // XOR-Shift constants
    static constexpr result_type a = 21;
    static constexpr result_type b = 35;
    static constexpr result_type c = 4;

    simple_rng(result_type seed) : state(seed ? seed : 88172645463325252ULL) {}

    // Generates a floating-point number in [0, 1)
    float generate() {
        state ^= state >> a;
        state ^= state << b;
        state ^= state >> c;
        return state * 5.421010862427522e-20f; // 2^-64
    }

private:
    result_type state;
};

static double random_double(int seed){
    simple_rng rng(seed);
    return rng.generate();
}

static vec3 randomVec(int seed){
    vec3 v(random_double(seed * 1e10), random_double(seed * 2.3e10), random_double(seed * 4.434e10));
}

static vec3 randomInUnitSphere() {
    vec3 p;
    int seed = 123123;
    while (true){
        simple_rng rng(seed);
        seed = rng.generate();
        p = randomVec(seed);
        if (p.length_squared() < 1)
            break;
    }
    return (p / p.length());

}

inline vec3 randomOnHemiSphere(const vec3& normal) {
    vec3 on_unit_sphere = randomInUnitSphere();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

class RenderKernel {
public:
    RenderKernel(sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::device> framePtr,
                 sycl::accessor<sphere, 1, sycl::access_mode::read, sycl::access::target::device> spherePtr,
                 sycl::accessor<Camera, 1, sycl::access_mode::read, sycl::access::target::device> cameraPtr,
                 int width, int height, int numSpheres) :
            m_framePtr(framePtr), m_spherePtr(spherePtr), m_cameraPtr(cameraPtr) {
            this->width = width;
            this->height = height;
            this->numSpheres = numSpheres;
        /* initialize accessors */ }
    int width, height, numSpheres;
    void operator()(sycl::nd_item<2> item) const { // Marked as const
        // get our Ids
        const auto x_coord = item.get_global_id(0);
        const auto y_coord = item.get_global_id(1);
        // map the 2D indices to a single linear, 1D index
        const auto pixel_index = y_coord * width + x_coord;

        int samplesPerPixel = 100;
        vec3 pixelColor(0, 0, 0);
        for (int sample = 0; sample < samplesPerPixel; ++sample){
            simple_rng rng(pixel_index * samplesPerPixel + sample); // Ensure each sample has a unique seed
            real_t u = (x_coord + rng.generate()) / width;
            real_t v = (y_coord + rng.generate()) / height;

            //ray r = get_ray(x_coord, y_coord, m_cameraPtr, (pixel_index * samplesPerPixel + sample));
            //pixelColor += ray_color(r, m_spherePtr);
        }

        pixelColor /= samplesPerPixel;

        m_framePtr[pixel_index] = pixelColor;
    }

private:

    double hit_sphere(const point3& center, double radius, const ray& r) const{
        vec3 oc = r.origin() - center;
        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius*radius;
        auto discriminant = half_b*half_b - a*c;

        if (discriminant < 0) {
            return -1.0;
        } else {
            return (-half_b - sqrt(discriminant) ) / a;
        }
    }

    vec3 ray_color(const ray& r, const sycl::global_ptr<sphere>& spheres) const{
        for (int i = 0; i < numSpheres; i++){
            auto t = hit_sphere(spheres[i].center, spheres[i].radius, r);
            if (t > 0.0) {
                vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
                return 0.5*vec3(N.x()+1, N.y()+1, N.z()+1);
            }
        }
        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0);
    }


    ray get_ray(real_t u, real_t v, const sycl::global_ptr<Camera>& camera, int idx) const {
        auto pixel_center = camera->pixel00Loc + (u * camera->pixelDeltaU) + (v * camera->pixelDeltaV);
        auto pixel_sample = pixel_center + pixel_sample_square(idx, camera);

        auto ray_origin = camera->center;
        auto ray_direction = pixel_sample - ray_origin;

        return { ray_origin, ray_direction};

    }

    vec3 pixel_sample_square(int idx, const sycl::global_ptr<Camera>& camera) const {
        // Returns a random point in the square surrounding a pixel at the origin.
        simple_rng rng(idx);
        rng.generate();
        auto px = -0.5 + rng.generate();
        auto py = -0.5 + rng.generate();
        return (px * camera->pixelDeltaU) + (py * camera->pixelDeltaV);
    }

    /* accessor objects */
    sycl::accessor<vec3, 1, sycl::access::mode::write, sycl::access::target::device> m_framePtr;
    sycl::accessor<sphere, 1, sycl::access::mode::read, sycl::access::target::device> m_spherePtr;
    sycl::accessor<Camera, 1, sycl::access::mode::read, sycl::access::target::device> m_cameraPtr;
};

#endif //SYCL_APP_KERNELS_H
