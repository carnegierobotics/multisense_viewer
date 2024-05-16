//
// Created by magnus on 4/10/24.
//

#include "SYCLRayTracer.h"


#include <fstream>
#include "RT_IN_ONE_WEEKEND/kernels.h"
#include "RT_IN_ONE_WEEKEND/camera.h"
#include "RT_IN_ONE_WEEKEND/sphere.h"


void SYCLRayTracer::save_image(const std::string& filename, uint32_t width, uint32_t height) {
    std::ofstream file(filename);

    if (!file) {
        std::cerr << "Failed to open the file for writing.\n";
        return;
    }

    file << "P3\n" << width << " " << height << "\n255\n";
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            auto pixel_index = y * width + x;
            int r = static_cast<int>(255.99 * fb[pixel_index].x());
            int g = static_cast<int>(255.99 * fb[pixel_index].y());
            int b = static_cast<int>(255.99 * fb[pixel_index].z());
            file << r << " " << g << " " << b << "\n";
        }
    }

    file.close();
    std::cout << "Image saved to " << filename << std::endl;
}

void SYCLRayTracer::render(int width, int height, int num_spheres, sycl::queue &queue, vec3 *fb_data, const sphere *spheres, Camera* camera) {
    auto num_pixels = width * height;
    auto frame_buf = sycl::buffer<vec3, 1>(fb_data, sycl::range<1>(num_pixels));
    sycl::buffer<sphere> spheres_buf(spheres, sycl::range<1>(num_spheres));
    sycl::buffer<Camera> cameraBuffer(camera, sycl::range<1>(1));

    static constexpr auto TileX = 8;
    static constexpr auto TileY = 8;
    static constexpr auto TileZ = 1;

    // submit command group on device
    queue.submit([&](sycl::handler &cgh) {
        // get memory access
        auto framePtr = frame_buf.get_access<sycl::access::mode::write>(cgh);
        auto spherePtr = spheres_buf.get_access<sycl::access::mode::read>(cgh);
        auto cameraPtr = cameraBuffer.get_access<sycl::access::mode::read>(cgh);
        // setup kernel index space
        const auto global = sycl::range<2>(width, height);
        const auto local = sycl::range<2>(TileX, TileY);
        const auto index_space = sycl::nd_range<2>(global, local);
        // construct kernel functor
        const auto render_k = RenderKernel(framePtr, spherePtr, cameraPtr, width, height, num_spheres);
        // execute kernel
        cgh.parallel_for(index_space, render_k);
    });
}


SYCLRayTracer::SYCLRayTracer(int width, int height) {
    // frame buffer dimensions

    auto num_pixels = width * height;

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (static_cast<double>(width)/height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / width;
    auto pixel_delta_v = viewport_v / height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                               - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    Camera camera;
    camera.pixel00Loc = pixel00_loc;
    camera.pixelDeltaU = pixel_delta_u;
    camera.pixelDeltaV = pixel_delta_v;
    camera.upperLeft = viewport_upper_left;
    camera.center = camera_center;

    // select the SYCL accelerator (i.e Intel GPU for this machine)
    auto queue = sycl::queue(sycl::gpu_selector_v);

    constexpr int samplesPerPixel = 5;

    fb.resize(num_pixels);

    constexpr auto num_spheres = 2;
    std::vector<sphere> spheres;
    spheres.emplace_back(vec3(0.0, 0.0, -1.0), 0.5);      // (small) center sphere
    spheres.emplace_back(sphere(vec3(0.0, -100.5, -1.0), 100.0)); // (large) ground sphere

    // run the SYCL render kenel
    render(width, height, num_spheres, queue, fb.data(), spheres.data(), &camera);

    // save the pixel data as an image file
    //save_image(fb.data(), "../output.ppm", width, height);
//
    //for (int i = 0; i < 25; ++i){
    //    simple_rng rng(((i + 1) * 1e10));
//
    //    printf("%f, %f, %f\n", rng.generate(), rng.generate(), rng.generate());
    //}


}

std::vector<SYCLRayTracer::Pixel> SYCLRayTracer::get_image_8bit(uint32_t width, uint32_t height) {
    std::vector<Pixel> data;
    data.resize(width * height);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            Pixel p{};
            auto pixel_index = y * width + x;
            p.r = static_cast<uint8_t>(255.99 * fb[pixel_index].x());
            p.g = static_cast<uint8_t>(255.99 * fb[pixel_index].y());
            p.b = static_cast<uint8_t>(255.99 * fb[pixel_index].z());
            p.a = 255;
            data[pixel_index] = p;
        }
    }
    return data;

}

