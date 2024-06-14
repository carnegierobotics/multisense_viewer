//
// Created by magnus on 5/17/24.
//

#include "GaussianRenderer.h"


#include "Viewer/Tools/Utils.h"
#include <filesystem>
#include <tinyply.h>
#include <glm/gtc/type_ptr.hpp>
#include <random>

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

#define BLOCK_X 16
#define BLOCK_Y 16

namespace VkRender {

    uint as_uint(const float x) {
        return *(uint *) &x;
    }

    float as_float(const uint x) {
        return *(float *) &x;
    }

    float half_to_float(
            const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint e = (x & 0x7C00) >> 10; // exponent
        const uint m = (x & 0x03FF) << 13; // mantissa
        const uint v = as_uint((float) m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
        return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) | ((e == 0) & (m != 0)) *
                                                                                ((v - 37) << 23 | ((m << (150 - v)) &
                                                                                                   0x007FE000))); // sign : normalized : denormalized
    }

    ushort float_to_half(
            const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
        const uint e = (b & 0x7F800000) >> 23; // exponent
        const uint m = b &
                       0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
        return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
               ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
               (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
    }

    //Test for correctness
    bool ValidationTest(std::vector<uint32_t> &keys, uint32_t size, uint32_t testIterations) {
        printf("Beginning VALIDATION tests at size %u and %u iterations. \n", size, testIterations);
        int testsPassed = 0;
        for (uint32_t i = 1; i <= testIterations; ++i) {
            bool isCorrect = true;
            for (uint32_t k = 1; k < keys.size(); ++k) {
                auto kBig = keys[k];
                auto kSmall = keys[k - 1];
                if (kBig < kSmall) {
                    isCorrect = false;
                    break;
                }
            }
            if (isCorrect)
                testsPassed++;
            else
                printf("Test iteration %d failed. \n", i);
        }
        printf("%d/%d tests passed.\n", testsPassed, testIterations);

        return testsPassed == testIterations;
    }


    uint8_t *GaussianRenderer::getImage() {
        return m_image;
    }

    uint32_t GaussianRenderer::getImageSize() {
        return m_initInfo.imageSize;
    }


    void GaussianRenderer::setup(const VkRender::AbstractRenderer::InitializeInfo &initInfo) {
        m_initInfo = initInfo;
        auto cpuSelector = [](const sycl::device &dev) {
            if (dev.is_cpu()) {
                return 1; // Positive value to prefer GPU devices
            } else {
                return -1; // Negative value to reject non-GPU devices
            }
        };    // Define a callable device selector using a lambda
        auto gpuSelector = [](const sycl::device &dev) {
            if (dev.is_gpu()) {
                return 1; // Positive value to prefer GPU devices
            } else {
                return -1; // Negative value to reject non-GPU devices
            }
        };

        try {
            // Create a queue using the CPU device selector
            queue = sycl::queue(gpuSelector, sycl::property::queue::in_order());
            // Use the queue for your computation
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("GPU device not found");
            Log::Logger::getInstance()->info("Falling back to default device selector");
            // Fallback to default device selector
            queue = sycl::queue(sycl::property::queue::in_order());
        }
        Log::Logger::getInstance()->info("Selected Device {}",
                                         queue.get_device().get_info<sycl::info::device::name>().c_str());

        gs = loadFromFile(Utils::getModelsPath() / "3dgs" / "coordinates.ply", 1);
        //gs = loadFromFile("/home/magnus/crl/multisense_viewer/3dgs_insect.ply", 1);
        loadedPly = false;
        setupBuffers(initInfo.camera);
        //simpleRasterizer(camera, false);
    }

    void GaussianRenderer::setupBuffers(const VkRender::Camera *camera) {
        Log::Logger::getInstance()->info("Loaded {} Gaussians", gs.getSize());
        positionBuffer = {gs.positions.data(), sycl::range<1>(gs.getSize())};
        scalesBuffer = {gs.scales.data(), sycl::range<1>(gs.getSize())};
        quaternionBuffer = {gs.quats.data(), sycl::range<1>(gs.getSize())};
        opacityBuffer = {gs.opacities.data(), sycl::range<1>(gs.getSize())};
        sphericalHarmonicsBuffer = {gs.sphericalHarmonics.data(), sycl::range<1>(gs.sphericalHarmonics.size())};

        // Create a buffer to store the resulting 2D covariance vectors
        covariance2DBuffer = {sycl::range<1>(gs.getSize())};
        conicBuffer = {sycl::range<1>(gs.getSize())};
        screenPosBuffer = sycl::buffer<glm::vec3, 1>{sycl::range<1>(gs.getSize())};
        covarianceBuffer = {sycl::range<1>(gs.getSize())};
        colorOutputBuffer = {sycl::range<1>(gs.getSize())};
        activeGSBuffer = {sycl::range<1>(gs.getSize())};
        imageBuffer = {sycl::range<3>(camera->m_height, camera->m_width, 4)};
        imageBuffer2 = {sycl::range<3>(camera->m_height, camera->m_width, 4)};
        width = camera->m_width;
        height = camera->m_height;


        keysBuffer = sycl::buffer<uint32_t, 1>((1 << 21)); // Adjust size later
        valuesBuffer = sycl::buffer<uint32_t, 1>((1 << 21));  // Adjust size later

        pointsBuffer = sycl::buffer<GaussianPoint, 1>(sycl::range<1>(gs.getSize()));
        numTilesTouchedBuffer = sycl::buffer<uint32_t, 1>(sycl::range<1>(gs.getSize()));
        numTilesTouchedInclusiveSumBuffer = sycl::buffer<uint32_t, 1>(sycl::range<1>(gs.getSize()));

    }

    glm::mat3 computeCov3D(const glm::vec3 &scale, const glm::quat &q) {

        glm::mat3 S(0.f);
        S[0][0] = scale.x;
        S[1][1] = scale.y;
        S[2][2] = scale.z;
        glm::mat3 R = glm::mat3_cast(q);
        glm::mat3 St = glm::transpose(S);
        glm::mat3 Rt = glm::transpose(R);
        glm::mat3 Sigma = R * S * St * Rt;
        return Sigma;

    }

    glm::vec3 computeCov2D(const glm::vec4 &pView,
                           const glm::mat3 &cov3D, const glm::mat4 &viewMat,
                           const GaussianRenderer::CameraParams &camera,
                           bool debug = false) {
        glm::vec4 t = pView;
        const float limx = 1.3f * camera.tanFovX;
        const float limy = 1.3f * camera.tanFovY;
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
        t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

        float l = glm::length(pView);
        glm::mat3 J = glm::mat3(camera.focalY / t.z, 0.0f, 0.0f,
                                0.0f, camera.focalY / t.z, 0.0f,
                                -(camera.focalY * t.x) / (t.z * t.z), -(camera.focalY * t.y) / (t.z * t.z), 0.0f);

        auto W = glm::mat3(viewMat);
        glm::mat3 T = J * W;
        glm::mat3 cov = T * cov3D * glm::transpose(T);

        cov[0][0] += 0.3f;
        cov[1][1] += 0.3f;
        return {cov[0][0], cov[1][0], cov[1][1]};
    }

    void
    getRect(const glm::vec2 p, int max_radius, glm::vec2 &rect_min, glm::vec2 &rect_max,
            glm::vec3 grid = glm::vec3(0.0f)) {
        rect_min = {
                std::min(grid.x, std::max(0.0f, ((p.x - max_radius) / BLOCK_X))),
                std::min(grid.y, std::max(0.0f, ((p.y - max_radius) / BLOCK_Y)))
        };
        rect_max = glm::vec2(
                std::min(grid.x, std::max(0.0f, ((p.x + max_radius + BLOCK_X - 1.0f) / BLOCK_X))),
                std::min(grid.y, std::max(0.0f, ((p.y + max_radius + BLOCK_Y - 1.0f) / BLOCK_Y)))
        );
    }

    void getRect(const glm::vec2 p, int max_radius, glm::ivec2 &rect_min, glm::ivec2 &rect_max,
                 glm::vec3 grid = glm::vec3(0.0f)) {
        rect_min = {
                std::min(grid.x, std::max(0.0f, ((p.x - max_radius) / BLOCK_X))),
                std::min(grid.y, std::max(0.0f, ((p.y - max_radius) / BLOCK_Y)))
        };
        rect_max = glm::vec2(
                std::min(grid.x, std::max(0.0f, ((p.x + max_radius + BLOCK_X - 1.0f) / BLOCK_X))),
                std::min(grid.y, std::max(0.0f, ((p.y + max_radius + BLOCK_Y - 1.0f) / BLOCK_Y)))
        );
    }

    template<typename T>
    T my_ceil(T x) {
        if (x == static_cast<int>(x)) {
            return x;
        } else {
            return static_cast<int>(x) + ((x > 0) ? 1 : 0);
        }
    }

// Function to calculate the tile ID for a given point
    uint32_t calculateTileID(float x, float y, uint32_t imageWidth, uint32_t tileWidth, uint32_t tileHeight) {
        uint32_t numTilesPerRow = imageWidth / tileWidth;
        uint32_t tileRow = static_cast<uint32_t>(std::floor(y / tileHeight));
        uint32_t tileCol = static_cast<uint32_t>(std::floor(x / tileWidth));
        return tileRow * numTilesPerRow + tileCol;
    }


    void printDurations(
            const std::chrono::duration<double> &durationRasterization,
            const std::chrono::duration<double> &durationIdentification,
            const std::chrono::duration<double> &durationSorting,
            const std::chrono::duration<double> &durationAccumulation,
            const std::chrono::duration<double> &durationPreprocess,
            const std::chrono::duration<double> &durationInclusiveSum,
            const std::chrono::duration<double> &durationTotal) {
        // Convert durations to milliseconds
        auto durationRasterizationMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationRasterization);
        auto durationIdentificationMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationIdentification);
        auto durationSortingMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationSorting);
        auto durationAccumulationMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationAccumulation);
        auto durationPreprocessUs = std::chrono::duration_cast<std::chrono::microseconds>(durationPreprocess);
        auto durationInclusiveSumMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationInclusiveSum);
        auto durationTotalMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationTotal);

        // Create an output string stream to format the output
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3); // Set fixed-point notation and precision

        oss << "Total Duration: " << durationTotalMs.count() << " ms\n";
        oss << "Rasterization: " << durationRasterizationMs.count() << " ms\n";
        oss << "Identification: " << durationIdentificationMs.count() << " ms\n";
        oss << "Sorting: " << durationSortingMs.count() << " ms\n";
        oss << "Accumulation: " << durationAccumulationMs.count() << " ms\n";
        oss << "Preprocessing: " << durationPreprocessUs.count() << " us\n";
        oss << "Inclusive Scan: " << durationInclusiveSumMs.count() << " ms\n\n";

        // Print the formatted string
        std::cout << oss.str();
    }


    struct ImageState {
        std::vector<glm::ivec2> ranges;
    };

    void identifyTileRanges(int L, const std::vector<uint32_t> &point_list_keys, ImageState &imgState) {
        // Initialize ranges with -1 to indicate uninitialized ranges
        imgState.ranges.resize(L, glm::ivec2(-1, -1));

        for (int idx = 0; idx < point_list_keys.size(); ++idx) {
            uint32_t key = point_list_keys[idx];
            uint16_t currentTile = key >> 16;
            if (currentTile >= 3600)
                continue;

            if (idx == 0) {
                imgState.ranges[currentTile].x = 0;
            } else {
                uint16_t prevtile = point_list_keys[idx - 1] >> 16;
                if (currentTile != prevtile) {
                    imgState.ranges[prevtile].y = idx;
                    imgState.ranges[currentTile].x = idx;
                }
            }

            if (idx == point_list_keys.size() - 1) {
                imgState.ranges[currentTile].y = point_list_keys.size();
            }
        }
    }

    void GaussianRenderer::render(const AbstractRenderer::RenderInfo &info, const VkRender::RenderUtils *renderUtils) {
        auto *camera = info.camera;
        auto params = getHtanfovxyFocal(camera->m_Fov, camera->m_height, camera->m_width);
        glm::mat4 viewMatrix = camera->matrices.view;
        glm::mat4 projectionMatrix = camera->matrices.perspective;
        glm::vec3 camPos = camera->pose.pos;

        // Flip the second row of the projection matrix
        projectionMatrix[1] = -projectionMatrix[1];

        const uint32_t imageWidth = width;
        const uint32_t imageHeight = height;
        const uint32_t tileWidth = 16;
        const uint32_t tileHeight = 16;
        glm::vec3 tileGrid((imageWidth + BLOCK_X - 1) / BLOCK_X, (imageHeight + BLOCK_Y - 1) / BLOCK_Y, 1);

        uint32_t gridSize = tileGrid.x * tileGrid.y;


        // Start timing
        auto startPreprocess = std::chrono::high_resolution_clock::now();
        {
            queue.submit([&](sycl::handler &h) {
                auto scales = scalesBuffer.get_access<sycl::access::mode::read>(h);
                auto quaternions = quaternionBuffer.get_access<sycl::access::mode::read>(h);
                auto positions = positionBuffer.get_access<sycl::access::mode::read>(h);
                auto shs = sphericalHarmonicsBuffer.get_access<sycl::access::mode::read>(h);
                auto opacitiesAccess = opacityBuffer.get_access<sycl::access::mode::read>(h);

                auto pointsBufferAccess = pointsBuffer.get_access<sycl::access::mode::write>(h);
                auto numTilesTouchedAccess = numTilesTouchedBuffer.get_access<sycl::access::mode::write>(h);

                uint32_t shDim = gs.getShDim();

                h.parallel_for(sycl::range<1>(gs.getSize()), [=](sycl::id<1> idx) {

                    numTilesTouchedAccess[idx] = 0;

                    glm::vec3 scale = scales[idx];
                    glm::quat q = quaternions[idx];
                    glm::vec3 position = positions[idx];

                    glm::vec4 posView = viewMatrix * glm::vec4(position, 1.0f);
                    glm::vec4 posClip = projectionMatrix * posView;
                    glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;

                    glm::vec3 threshold(1.0f);
                    // Manually compute absolute values
                    glm::vec3 pos_screen_abs = glm::vec3(
                            std::abs(posNDC.x),
                            std::abs(posNDC.y),
                            std::abs(posNDC.z)
                    );
                    if (glm::any(glm::greaterThan(pos_screen_abs, threshold))) {
                        //screenPos[idx] = glm::vec4(-100.0f, -100.0f, -100.0f, 1.0f);
                        //sycl::ext::oneapi::experimental::printf("Culled: x = %f, y = %f, z = %f\n", position.x, position.y, position.z);
                        //activeGSAccess[idx] = false;
                        return;
                    }
                    float pixPosX = ((posNDC.x + 1.0f) * imageWidth - 1.0f) * 0.5f;
                    float pixPosY = ((posNDC.y + 1.0f) * imageHeight - 1.0f) * 0.5f;
                    auto screenPosPoint = glm::vec3(pixPosX, pixPosY, posNDC.z);

                    glm::mat3 cov3D = computeCov3D(scale, q);
                    glm::vec3 cov2D = computeCov2D(posView, cov3D, viewMatrix, params, false);

                    // Invert covariance (EWA)
                    float determinant = cov2D.x * cov2D.z - (cov2D.y * cov2D.y);
                    if (determinant != 0) {
                        float invDeterminant = 1 / determinant;

                        // Compute extent in screen space (by finding eigenvalues of
                        // 2D covariance matrix). Use extent to compute a bounding rectangle
                        // of screen-space tiles that this Gaussian overlaps with. Quit if
                        // rectangle covers 0 tiles.

                        float mid = 0.5f * (cov2D.x + cov2D.z);
                        float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - determinant));
                        float lambda2 = mid - std::sqrt(std::max(0.1f, mid * mid - determinant));
                        float my_radius = my_ceil(3.f * std::sqrt(std::max(lambda1, lambda2)));
                        glm::vec2 rect_min, rect_max;
                        getRect(screenPosPoint, my_radius, rect_min, rect_max, tileGrid);
                        if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
                            return;

                        glm::vec3 dir = glm::normalize(position - camPos);
                        glm::vec3 color =
                                SH_C0 * glm::vec3(shs[idx * shDim + 0], shs[idx * shDim + 1], shs[idx * shDim + 2]);
                        color += 0.5f;

                        auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant,
                                               cov2D.x * invDeterminant);
                        pointsBufferAccess[idx].depth = posNDC.z;
                        pointsBufferAccess[idx].radius = my_radius;
                        pointsBufferAccess[idx].conic = conic;
                        pointsBufferAccess[idx].screenPos = screenPosPoint;
                        pointsBufferAccess[idx].color = color;
                        pointsBufferAccess[idx].opacityBuffer = opacitiesAccess[idx];
                        // How many tiles we access
                        // rect_min/max are in tile space
                        numTilesTouchedAccess[idx] = static_cast<int>((rect_max.y - rect_min.y) *
                                                                      (rect_max.x - rect_min.x));
                    }
                });
            }).wait();
            // Stop timing
            auto endPreprocess = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationPreprocess = endPreprocess - startPreprocess;

            auto startInclusiveSum = std::chrono::high_resolution_clock::now();

            //parallel_inclusive_scan(queue, numTilesTouchedBuffer, numTilesTouchedInclusiveSumBuffer, numTilesTouchedBuffer.size());
            uint32_t numRendered;

            /*
            queue.submit([&](sycl::handler &h) {
                // Accessors for buffers
                uint32_t n = numTilesTouchedBuffer.size();
                auto g_odata_acc = numTilesTouchedInclusiveSumBuffer.get_access<sycl::access::mode::write>(h);
                auto g_idata_acc = numTilesTouchedBuffer.get_access<sycl::access::mode::read>(h);

                // Local memory for shared data
                sycl::local_accessor<float, 1> temp(n * 2, h);

                h.parallel_for<class scan_kernel>(sycl::nd_range<1>(n, n), [=](sycl::nd_item<1> item) {
                    int thid = item.get_local_id(0);
                    int pout = 0, pin = 1;

                    // Load input into shared memory
                    temp[pout * n + thid] = g_idata_acc[thid];
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int offset = 1; offset < n; offset *= 2) {
                        pout = 1 - pout; // swap double buffer indices
                        pin = 1 - pout;
                        if (thid >= offset)
                            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
                        else
                            temp[pout * n + thid] = temp[pin * n + thid];
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    g_odata_acc[thid] = temp[pout * n + thid]; // write output
                });
            }).wait();
            */
            const auto &numTilesTouchedHost = numTilesTouchedBuffer.get_host_access<>();
            std::vector<uint32_t> numTilesTouchedSum(numTilesTouchedBuffer.size());
            numTilesTouchedSum[0] = numTilesTouchedHost[0];
            for (size_t i = 1; i < numTilesTouchedHost.size(); ++i) {
                numTilesTouchedSum[i] = numTilesTouchedSum[i - 1] + numTilesTouchedHost[i];
            }

            numTilesTouchedInclusiveSumBuffer = sycl::buffer<uint32_t, 1>(numTilesTouchedSum.data(),
                                                                          numTilesTouchedSum.size());


            auto endInclusiveSum = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationInclusiveSum = endInclusiveSum - startInclusiveSum;

            numRendered = numTilesTouchedInclusiveSumBuffer.get_host_access().get_pointer()[gs.getSize() - 1];

            auto startAccumulation = std::chrono::high_resolution_clock::now();

            /*
            {
                std::vector<uint32_t> inclusiveSum(numTilesTouchedInclusiveSumBuffer.size());
                auto inclusiveSumAcc = numTilesTouchedInclusiveSumBuffer.get_host_access();
                for (size_t i = 0; i < inclusiveSumAcc.size(); ++i) {
                    inclusiveSum[i] = inclusiveSumAcc[i];
                }
                auto numRendered2 = inclusiveSum[inclusiveSum.size() - 1];

                const auto &numTilesTouchedHost = numTilesTouchedBuffer.get_host_access<>();
                for (size_t i = 1; i < numTilesTouchedHost.size(); ++i) {
                    numTilesTouchedHost[i] += numTilesTouchedHost[i - 1];
                }
                int k2 = 0;
            }
            */

            {
                std::vector<uint32_t> gaussian_keys_unsorted;  // Adjust size later
                std::vector<uint32_t> gaussian_values_unsorted;  // Adjust size later

                gaussian_keys_unsorted.resize(numRendered);
                gaussian_values_unsorted.resize(numRendered);

                const auto &gaussianBufferHost = pointsBuffer.get_host_access<>();
                auto inclusiveSumAcc = numTilesTouchedInclusiveSumBuffer.get_host_access();
                for (int idx = 0; idx < gs.getSize(); ++idx) {
                    const auto &gaussian = gaussianBufferHost[idx];

                    // Generate no key/value pair for invisible Gaussians
                    if (gaussian.radius > 0) {
                        // Find this Gaussian's offset in buffer for writing keys/values.
                        uint32_t off = (idx == 0) ? 0 : inclusiveSumAcc[idx - 1];
                        glm::ivec2 rect_min, rect_max;

                        getRect(gaussian.screenPos, gaussian.radius, rect_min, rect_max, tileGrid);

                        // For each tile that the bounding rect overlaps, emit a key/value pair.
                        for (int y = rect_min.y; y < rect_max.y; ++y) {
                            for (int x = rect_min.x; x < rect_max.x; ++x) {
                                if (off < numRendered) {
                                    uint32_t key = y * tileGrid.x + x;
                                    if (key >= gridSize) {
                                        key = 0;
                                    }
                                    key <<= 16;
                                    uint16_t half = float_to_half(gaussian.depth);
                                    key |= half;
                                    gaussian_keys_unsorted[off] = key;
                                    gaussian_values_unsorted[off] = static_cast<uint32_t>(idx);
                                } else {
                                    int notGreat = 1;
                                }
                                ++off;
                            }
                        }
                    }
                }
            }


            keysBuffer = sycl::buffer<uint32_t, 1>(numRendered); // Adjust size later
            valuesBuffer = sycl::buffer<uint32_t, 1>(numRendered);  // Adjust size later

            queue.submit([&](sycl::handler &cgh) {
                auto keysAcc = keysBuffer.get_access<sycl::access::mode::write>(cgh);
                auto valuesAcc = valuesBuffer.get_access<sycl::access::mode::write>(cgh);

                auto inclusiveSumAccess = numTilesTouchedInclusiveSumBuffer.get_access<sycl::access::mode::read>(cgh);
                auto gaussianAccess = pointsBuffer.get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for<class GaussianProcessing>(sycl::range<1>(gs.getSize()), [=](sycl::id<1> idx) {
                    uint32_t i = idx.get(0);
                    const auto &gaussian = gaussianAccess[i];

                    if (gaussian.radius > 0) {
                        uint32_t off = (i == 0) ? 0 : inclusiveSumAccess[i - 1];
                        glm::ivec2 rect_min, rect_max;
                        getRect(gaussian.screenPos, gaussian.radius, rect_min, rect_max, tileGrid);
                        for (int y = rect_min.y; y < rect_max.y; ++y) {
                            for (int x = rect_min.x; x < rect_max.x; ++x) {
                                if (off >= numRendered) {
                                    break;
                                }
                                uint32_t key = static_cast<uint16_t>(y) * static_cast<uint16_t>(tileGrid.x) + x;
                                if (key >= gridSize) {
                                    key = 0;
                                }
                                key <<= 16;
                                uint16_t half = float_to_half(gaussian.depth);
                                key |= half;
                                keysAcc[off] = key;
                                valuesAcc[off] = i;
                                ++off;
                            }
                        }
                    }
                });
            }).wait();
            auto endAccumulation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationAccumulation = endAccumulation - startAccumulation;


            /*
            if (numTilesTouchedInclusiveSumBuffer.get_host_access()[numTilesTouchedInclusiveSumBuffer.size()] <= 0) {
                auto hostImageAccessor = imageBuffer.get_host_access();
                m_image = hostImageAccessor.get_pointer();
                return;
            }
            */


            auto startSorting = std::chrono::high_resolution_clock::now();
            radixSorter = std::make_unique<crl::RadixSorter>(queue, numRendered);

            radixSorter->performOneSweep(numRendered, keysBuffer, valuesBuffer);
            queue.wait();

            {
                std::vector<uint32_t> keysHost(keysBuffer.size());
                std::vector<uint32_t> valuesHost(valuesBuffer.size());
                // Copy data back to host
                auto keysHostAcc = keysBuffer.get_host_access();
                uint64_t sum = 0;
                for (size_t i = 0; i < keysBuffer.size(); ++i) {
                    keysHost[i] = keysHostAcc[i];
                    sum += keysHost[i];
                }
                /*
                auto valuesHostAcc = valuesBuffer.get_host_access();
                for (size_t i = 0; i < valuesHostAcc.size(); ++i) {
                    valuesHost[i] = valuesHostAcc[i];
                    if (valuesHost[i] >= 3600) {
                        printf("Error %zu, %d\n", i, valuesHost[i]);
                    }
                }
                */
                if (ValidationTest(keysHost, keysHost.size(), 2)) {
                    int yes = 1;
                } else {
                    int no = 1;
                }
            }


            auto endSorting = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationSorting = endSorting - startSorting;

            auto startIdentification = std::chrono::high_resolution_clock::now();

            sycl::buffer<glm::ivec2, 1> rangesBuffer(tileGrid.x * tileGrid.y);

            // IdentifyTileRanges
            {
                queue.submit([&](sycl::handler &cgh) {
                    auto keysAcc = keysBuffer.get_access<sycl::access::mode::read>(cgh);
                    auto rangesAcc = rangesBuffer.get_access<sycl::access::mode::read_write>(cgh);

                    cgh.parallel_for<class TileRanges>(sycl::range<1>(rangesBuffer.size()), [=](sycl::id<1> idx) {
                        int i = idx[0];
                        rangesAcc[i] = glm::ivec2(-1, -1);
                        uint32_t key = keysAcc[i];
                        uint16_t currTile = key >> 16;
                        if (currTile < 3600) {
                            if (i == 0) {
                                rangesAcc[currTile].x = 0;
                            } else {
                                uint16_t prevTile = keysAcc[i - 1] >> 16;
                                if (currTile != prevTile) {
                                    rangesAcc[prevTile].y = i;
                                    rangesAcc[currTile].x = i;
                                }
                            }

                            if (i == keysAcc.size() - 1) {
                                rangesAcc[currTile].y = keysAcc.size();
                            }
                        }
                    });
                }).wait();
            }
            auto endIdentification = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationIdentification = endIdentification - startIdentification;

            ImageState imgState;
            {
                std::vector<uint32_t> keys(keysBuffer.size());
                auto keysAcc = keysBuffer.get_host_access();
                for (size_t i = 0; i < keysAcc.size(); ++i) {
                    keys[i] = keysAcc[i];
                }
                identifyTileRanges(tileGrid.x * tileGrid.y, keys, imgState);
                rangesBuffer = sycl::buffer<glm::ivec2, 1>{imgState.ranges.data(), imgState.ranges.size()};
            }
            // Compute the global work size ensuring it is a multiple of the local work size
            sycl::range<2> localWorkSize(tileHeight, tileWidth);
            size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
            size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
            sycl::range<2> globalWorkSize(globalHeight, globalWidth);
            uint32_t horizontal_blocks = (imageWidth + tileWidth - 1) / tileWidth;

            auto startRasterize = std::chrono::high_resolution_clock::now();
            queue.submit([&](sycl::handler &h) {
                auto rangesBufferAccess = rangesBuffer.get_access<sycl::access::mode::read>(h);
                auto gaussianBufferAccess = pointsBuffer.get_access<sycl::access::mode::read>(h);
                auto pointListBufferAccess = valuesBuffer.get_access<sycl::access::mode::read>(h);
                auto imageAccessor = imageBuffer.get_access<sycl::access::mode::write>(h);
                uint32_t numPoints = gs.getSize();
                h.parallel_for(sycl::nd_range<2>(globalWorkSize, localWorkSize), [=](sycl::nd_item<2> item) {
                    auto globalID = item.get_global_id(); // Get global indices of the work item
                    uint32_t row = globalID[0];
                    uint32_t col = globalID[1];


                    if (row < imageHeight && col < imageWidth) {
                        //uint32_t tileId = calculateTileID(row, col, imageWidth, tileWidth, tileHeight);
                        //uint32_t tileId = calculateTileID(row, col, imageWidth, tileWidth, tileHeight);
                        uint32_t groupRow = row / 16;
                        uint32_t groupCol = col / 16;
                        uint32_t tileId = groupRow * horizontal_blocks + groupCol;

                        // Ensure tileId is within bounds
                        if (tileId >= rangesBufferAccess.size()) {
                            sycl::ext::oneapi::experimental::printf(
                                    "TileId %u out of bounds (max %u ). groupRow %u, groupCol %u, horizontal_blocks %u, imageWidth %u \n",
                                    tileId, static_cast<uint32_t>(rangesBufferAccess.size() - 1), groupRow, groupCol,
                                    horizontal_blocks, imageWidth);
                            return;
                        }
                        //size_t tileId = group.get_group_id(1) * horizontal_blocks + group.get_group_id(0);
                        glm::ivec2 range = rangesBufferAccess[tileId];
                        // Initialize helper variables
                        float T = 1.0f;
                        float C[3] = {0};
                        if (range.x >= 0 && range.y >= 0) {

                            for (int listIndex = range.x; listIndex < range.y; ++listIndex) {
                                uint32_t index = pointListBufferAccess[listIndex];
                                const GaussianPoint &point = gaussianBufferAccess[index];
                                if (index >= numPoints || listIndex >= 3600)
                                    continue;
                                //sycl::ext::oneapi::experimental::printf("ListIndex: %u index: %u\n", listIndex, index);


                                // Perform processing on the point and update the image
                                // Example: Set the pixel to a specific value
                                glm::vec2 pos = point.screenPos;
                                // Calculate the exponent term
                                glm::vec2 diff = glm::vec2(col, row) - pos;
                                glm::vec3 c = point.conic;
                                glm::mat2 V(c.x, c.y, c.y, c.z);
                                float power = -0.5f * glm::dot(diff, V * diff);
                                if (power > 0.0f) {
                                    continue;
                                }
                                float alpha = std::min(0.99f, point.opacityBuffer * expf(power));
                                if (alpha < 1.0f / 255.0f)
                                    continue;
                                float test_T = T * (1 - alpha);
                                if (test_T < 0.0001f) {
                                    continue;
                                }
                                // Eq. (3) from 3D Gaussian splatting paper.
                                for (int ch = 0; ch < 3; ch++) {
                                    C[ch] += point.color[ch] * alpha * T;
                                }
                                T = test_T;
                            }
                        }
                        imageAccessor[row][col][0] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
                        imageAccessor[row][col][1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
                        imageAccessor[row][col][2] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
                        imageAccessor[row][col][3] = static_cast<uint8_t>(255.0f);
                    } // endif
                });
            }).wait();

            auto endRasterization = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationRasterization = endRasterization - startRasterize;

            std::chrono::duration<double> totalDuration = endRasterization - startPreprocess;

            printDurations(durationRasterization, durationIdentification, durationSorting, durationAccumulation,
                           durationPreprocess, durationInclusiveSum, totalDuration);
        }

        m_image = imageBuffer.get_host_access().get_pointer();
        // Use `image` as needed
    }

    GaussianRenderer::GaussianPoints GaussianRenderer::loadNaive() {
        GaussianRenderer::GaussianPoints data;
        auto unitQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        auto scale = glm::vec3(0.05f, 0.05f, 0.05f);
        auto pos
                = glm::vec3(0.0f, 0.0f, 0.0f);

        data.positions.emplace_back(pos);
        data.positions.emplace_back(pos + glm::vec3(1.0f, 0.0f, 0.0f));
        data.positions.emplace_back(pos + glm::vec3(0.0f, 1.0f, 0.0f));
        data.positions.emplace_back(pos + glm::vec3(0.0f, 0.0f, 1.0f));

        data.scales.emplace_back(glm::vec3(0.03f, 0.03f, 0.03f));
        data.scales.emplace_back(glm::vec3(0.5f, 0.03f, 0.03f));
        data.scales.emplace_back(glm::vec3(0.03f, 0.5f, 0.03f));
        data.scales.emplace_back(glm::vec3(0.03f, 0.03f, 0.5f));

        for (int i = 0; i < 4; ++i) {
            data.quats.emplace_back(unitQuat);
            data.opacities.emplace_back(1.0f);
        }

        std::vector<float> sphericalHarmonics = {
                1.0f, 0.0f, 1.0f,   // White
                1.0f, 0.0f, 0.0f,  // Red
                0.0f, 1.0f, 0.0f,  // Green
                0.0f, 0.0f, 1.0f,  // Blue
        };
        for (auto &sh: sphericalHarmonics)
            sh = (sh - 0.5f) / 0.28209f;

        data.sphericalHarmonics = sphericalHarmonics;
        data.shDim = 3;
        return data;

    }

    GaussianRenderer::GaussianPoints GaussianRenderer::loadFromFile(std::filesystem::path path, int downSampleRate) {
        GaussianPoints data;
        //auto plyFilePath = std::filesystem::path("/home/magnus/phd/SuGaR/output/refined_ply/0000/3dgs.ply");
        auto plyFilePath = std::filesystem::path(path);

// Open the PLY file
        std::ifstream ss(plyFilePath, std::ios::binary);
        if (!ss.is_open()) {
            throw std::runtime_error("Failed to open PLY file.");
        }

        tinyply::PlyFile file;
        file.parse_header(ss);

        std::shared_ptr<tinyply::PlyData> vertices, scales, quats, opacities, colors, harmonics;

        try { vertices = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { quats = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { opacities = file.request_properties_from_element("vertex", {"opacity"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
        // Request spherical harmonics properties
        std::vector<std::string> harmonics_properties;
        for (int i = 0; i < 45; ++i) {
            harmonics_properties.push_back("f_rest_" + std::to_string(i));
        }
        try { harmonics = file.request_properties_from_element("vertex", harmonics_properties); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


        file.read(ss);
        const size_t numVertices = vertices->count;

        // Process vertices
        if (vertices) {
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            std::vector<float> vertexBuffer(numVertices * 3);
            std::memcpy(vertexBuffer.data(), vertices->buffer.get(), numVerticesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.positions.emplace_back(vertexBuffer[i * 3], vertexBuffer[i * 3 + 1], vertexBuffer[i * 3 + 2]);
            }
        }

        // Process scales
        if (scales) {
            const size_t numScalesBytes = scales->buffer.size_bytes();
            std::vector<float> scaleBuffer(numVertices * 3);
            std::memcpy(scaleBuffer.data(), scales->buffer.get(), numScalesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.scales.emplace_back(expf(scaleBuffer[i * 3]), expf(scaleBuffer[i * 3 + 1]),
                                         expf(scaleBuffer[i * 3 + 2]));
            }
        }

        // Process quats
        if (quats) {
            const size_t numQuatsBytes = quats->buffer.size_bytes();
            std::vector<float> quatBuffer(numVertices * 4);
            std::memcpy(quatBuffer.data(), quats->buffer.get(), numQuatsBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.quats.emplace_back(quatBuffer[i * 4], quatBuffer[i * 4 + 1], quatBuffer[i * 4 + 2],
                                        quatBuffer[i * 4 + 3]);
            }
        }

        // Process opacities
        if (opacities) {
            const size_t numOpacitiesBytes = opacities->buffer.size_bytes();
            std::vector<float> opacityBuffer(numVertices);
            std::memcpy(opacityBuffer.data(), opacities->buffer.get(), numOpacitiesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                float opacity = opacityBuffer[i];
                opacity = 1.0f / (1.0f + expf(-opacity));
                data.opacities.push_back(opacity);
            }
        }

        // Process colors and spherical harmonics
        if (colors && harmonics) {
            const size_t numColorsBytes = colors->buffer.size_bytes();
            std::vector<float> colorBuffer(numVertices * 3);
            std::memcpy(colorBuffer.data(), colors->buffer.get(), numColorsBytes);

            const size_t numHarmonicsBytes = harmonics->buffer.size_bytes();
            std::vector<float> harmonicsBuffer(numVertices * harmonics_properties.size());
            std::memcpy(harmonicsBuffer.data(), harmonics->buffer.get(), numHarmonicsBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.sphericalHarmonics.push_back(colorBuffer[i * 3]);
                data.sphericalHarmonics.push_back(colorBuffer[i * 3 + 1]);
                data.sphericalHarmonics.push_back(colorBuffer[i * 3 + 2]);

                for (size_t j = 0; j < harmonics_properties.size(); ++j) {
                    data.sphericalHarmonics.push_back(harmonicsBuffer[i * harmonics_properties.size() + j]);
                }
            }

            data.shDim = 3 + harmonics_properties.size();
        }
        return data;
    }


}
