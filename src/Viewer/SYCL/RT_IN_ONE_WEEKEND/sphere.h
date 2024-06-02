//
// Created by magnus on 4/3/24.
//

#ifndef SYCL_APP_SPHERE_H
#define SYCL_APP_SPHERE_H
#include <sycl/sycl.hpp>
#include "vec3.h"
#include "ray.h"
#include "hittable.h"

using real_t = float;

class sphere : public hittable{
public:
    sphere(const vec3& centerPoint, real_t _radius) : center(centerPoint), radius(_radius) {}

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const {
        vec3 oc = r.origin() - center;
        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (-half_b + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }


    // geometry properties
    vec3 center;
    real_t radius;
};

#endif //SYCL_APP_SPHERE_H
