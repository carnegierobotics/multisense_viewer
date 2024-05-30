//
// Created by magnus on 5/17/24.
//

#include "GaussianRenderer.h"
#include "Viewer/Tools/Utils.h"

#include <filesystem>
#include <tinyply.h>
#include <glm/gtc/type_ptr.hpp>

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

GaussianRenderer::GaussianRenderer(const VkRender::Camera &camera) {
    // Define a callable device selector using a lambda
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
        queue = sycl::queue(gpuSelector);
        // Use the queue for your computation
    } catch (const sycl::exception &e) {
        Log::Logger::getInstance()->warning("GPU device not found");
        Log::Logger::getInstance()->info("Falling back to default device selector");
        // Fallback to default device selector
        queue = sycl::queue();
    }
    Log::Logger::getInstance()->info("Selected Device {}",
                                     queue.get_device().get_info<sycl::info::device::name>().c_str());

    gs = loadFromFile("../3dgs_coordinates.ply", 1);
    //gs = loadFromFile("/home/magnus/crl/multisense_viewer/3dgs_insect.ply", 1);
    loadedPly = false;
    setupBuffers(camera);

    //simpleRasterizer(camera, false);
}

void GaussianRenderer::setupBuffers(const VkRender::Camera &camera) {
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

    int texWidth, texHeight, texChannels;
    std::filesystem::path path = Utils::getTexturePath() / "moon.png";
    image = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!image) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return;
    }

    //sycl::buffer<uint8_t, 1> pngImageBuffer{image, sycl::range<1>(texWidth * texHeight * channels)};
    pngImageBuffer = {image, sycl::range<3>(texHeight, texWidth, 4)};
    imageBuffer = {sycl::range<3>(camera.m_height, camera.m_width, 4)};
    width = camera.m_width;
    height = camera.m_height;
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
                       const glm::mat3 &cov3D, const glm::mat4 &viewMat, const GaussianRenderer::CameraParams &camera,
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

    if (debug) {
        // Print matrices using printf
        sycl::ext::oneapi::experimental::printf("Length L: %f\n", l);

        sycl::ext::oneapi::experimental::printf("Matrix W:\n");
        for (int i = 0; i < 3; ++i) {
            sycl::ext::oneapi::experimental::printf("%f %f %f\n", W[i][0], W[i][1], W[i][2]);
        }
        sycl::ext::oneapi::experimental::printf("\n");

        sycl::ext::oneapi::experimental::printf("Matrix J:\n");
        for (int i = 0; i < 3; ++i) {
            sycl::ext::oneapi::experimental::printf("%f %f %f\n", J[i][0], J[i][1], J[i][2]);
        }
        sycl::ext::oneapi::experimental::printf("\n");

        sycl::ext::oneapi::experimental::printf("Matrix T:\n");
        for (int i = 0; i < 3; ++i) {
            sycl::ext::oneapi::experimental::printf("%f %f %f\n", T[i][0], T[i][1], T[i][2]);
        }
        sycl::ext::oneapi::experimental::printf("\n");

        sycl::ext::oneapi::experimental::printf("Matrix cov2D:\n");
        for (int i = 0; i < 3; ++i) {
            sycl::ext::oneapi::experimental::printf("%f %f %f\n", cov[i][0], cov[i][1], cov[i][2]);
        }
        sycl::ext::oneapi::experimental::printf("\n");

        sycl::ext::oneapi::experimental::printf("Matrix cov3D:\n");
        for (int i = 0; i < 3; ++i) {
            sycl::ext::oneapi::experimental::printf("%f %f %f\n", cov3D[i][0], cov3D[i][1], cov3D[i][2]);
        }

        sycl::ext::oneapi::experimental::printf("\n");
    }
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return {cov[0][0], cov[1][0], cov[1][1]};
}

void
getRect(const glm::vec2 p, int max_radius, glm::vec2 &rect_min, glm::vec2 &rect_max, glm::vec3 grid = glm::vec3(0.0f)) {
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

struct BinningState {
    std::vector<uint64_t> point_list_keys_unsorted;
    std::vector<uint64_t> point_list_keys;
    std::vector<uint32_t> point_list_unsorted;
    std::vector<uint32_t> point_list;
};

void duplicateWithKeys(
        int P,
        const GaussianRenderer::GaussianPoint *gaussians,
        const uint32_t *offsets,
        std::vector<uint64_t> &gaussian_keys_unsorted,
        std::vector<uint32_t> &gaussian_values_unsorted,
        const glm::vec3 &grid) {
    for (int idx = 0; idx < P; ++idx) {
        const auto &gaussian = gaussians[idx];

        // Generate no key/value pair for invisible Gaussians
        if (gaussian.radius > 0) {
            // Find this Gaussian's offset in buffer for writing keys/values.
            uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
            glm::ivec2 rect_min, rect_max;

            getRect(gaussian.screenPos, gaussian.radius, rect_min, rect_max, grid);

            // For each tile that the bounding rect overlaps, emit a key/value pair.
            for (int y = rect_min.y; y < rect_max.y; ++y) {
                for (int x = rect_min.x; x < rect_max.x; ++x) {
                    uint64_t key = static_cast<uint64_t>(y) * grid.x + x;
                    key <<= 32;
                    key |= *reinterpret_cast<const uint32_t *>(&gaussian.depth);
                    gaussian_keys_unsorted[off] = key;
                    gaussian_values_unsorted[off] = static_cast<uint32_t>(idx);
                    ++off;
                }
            }
        }
    }
}

struct ImgState {
    std::vector<uint32_t> ranges;
};

void identifyTileRanges(int num_rendered, const std::vector<uint64_t> &point_list_keys, std::vector<uint32_t> &ranges,
                        const glm::vec3 &tile_grid) {
    uint32_t num_tiles = tile_grid.x * tile_grid.y;

    // Initialize ranges to maximum value
    ranges.assign(num_tiles * 2, std::numeric_limits<uint32_t>::max());  // 2 entries per tile: start and end

    // Identify start and end ranges for each tile
    for (int i = 0; i < num_rendered; ++i) {
        uint64_t key = point_list_keys[i];
        uint32_t tile_id = key >> 32;

        if (ranges[tile_id * 2] == std::numeric_limits<uint32_t>::max()) {
            ranges[tile_id * 2] = i;  // Start range
        }
        ranges[tile_id * 2 + 1] = i + 1;  // End range (exclusive)
    }
}


// Function to process Gaussian points and update the image
void processGaussianPoints(GaussianRenderer::GaussianPoint *points, const std::vector<uint32_t> &ranges,
                           std::vector<std::vector<std::array<uint8_t, 4>>> &image, uint32_t imageWidth,
                           uint32_t imageHeight,
                           uint32_t tileWidth, uint32_t tileHeight) {
    uint32_t num_tiles = ranges.size() / 2;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t start = ranges[tile * 2];
        uint32_t end = ranges[tile * 2 + 1];

        if (start != std::numeric_limits<uint32_t>::max()) {
            for (uint32_t id = start; id < end; ++id) {
                const GaussianRenderer::GaussianPoint &point = points[id];
                glm::vec2 pos = point.screenPos;
                glm::vec2 diff;
                glm::vec3 c = point.conic;
                glm::mat2 V(c.x, c.y, c.y, c.z);
                for (uint32_t row = 0; row < imageHeight; ++row) {
                    for (uint32_t col = 0; col < imageWidth; ++col) {
                        if (row < imageHeight && col < imageWidth) {
                            uint32_t tileID = calculateTileID(col, row, imageWidth, tileWidth, tileHeight);
                            if (tileID == tile) {
                                diff = glm::vec2(col, row) - pos;
                                float power = -0.5f * glm::dot(diff, V * diff);
                                if (power > 0.0f) {
                                    continue;
                                }
                                float alpha = std::min(0.99f, point.opacityBuffer * expf(power));

                                if (alpha < 1.0f / 255.0f) {
                                    continue;
                                }

                                float T = 1.0f;
                                float C[3] = {0};
                                float test_T = T * (1 - alpha);
                                if (test_T < 0.0001f) {
                                    continue;
                                }
                                for (int ch = 0; ch < 3; ch++) {
                                    C[ch] += point.color[ch] * alpha * T;
                                }
                                T = test_T;

                                image[row][col][0] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
                                image[row][col][1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
                                image[row][col][2] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
                                image[row][col][3] = static_cast<uint8_t>(255.0f);
                            }
                        }
                    }
                }
            }
        }
    }
}


void GaussianRenderer::tileRasterizer(const VkRender::Camera &camera) {
    auto params = getHtanfovxyFocal(camera.m_Fov, camera.m_height, camera.m_width);
    glm::mat4 viewMatrix = camera.matrices.view;
    glm::mat4 projectionMatrix = camera.matrices.perspective;
    glm::vec3 camPos = camera.pose.pos;

    // Flip the second row of the projection matrix
    projectionMatrix[1] = -projectionMatrix[1];

    const size_t imageWidth = width;
    const size_t imageHeight = height;
    const size_t tileWidth = 16;
    const size_t tileHeight = 16;
    glm::vec3 tileGrid((imageWidth + BLOCK_X - 1) / BLOCK_X, (imageHeight + BLOCK_Y - 1) / BLOCK_Y, 1);

    sycl::buffer<GaussianPoint, 1> pointsBuffer(sycl::range<1>(gs.getSize()));
    sycl::buffer<uint32_t, 1> numTilesTouchedBuffer(sycl::range<1>(gs.getSize()));

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
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

            // Which tile does this gaussian belong to?
            uint32_t tileID = calculateTileID(pixPosX, pixPosY, imageWidth, tileWidth, tileHeight);
            //activeGSAccess[idx] = true;
            // Calculate which tile

            glm::mat3 cov3D = computeCov3D(scale, q);
            //covariances[idx] = cov3D;

            glm::vec3 cov2D = computeCov2D(posView, cov3D, viewMatrix, params, false);
            //covariances2D[idx] = cov2D;

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
                glm::vec3 color = SH_C0 * glm::vec3(shs[idx * shDim + 0], shs[idx * shDim + 1], shs[idx * shDim + 2]);
                color += 0.5f;

                auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant, cov2D.x * invDeterminant);
                //screenPos[idx] = glm::vec4(screenPosPoint, 1.0f);
                //conicAccess[idx] = conic;
                //colorOutput[idx] = color;

                pointsBufferAccess[idx].depth = posNDC.z;
                pointsBufferAccess[idx].radius = my_radius;
                pointsBufferAccess[idx].conic = conic;
                pointsBufferAccess[idx].screenPos = screenPosPoint;
                pointsBufferAccess[idx].color = color;
                pointsBufferAccess[idx].opacityBuffer = opacitiesAccess[idx];

                numTilesTouchedAccess[idx] = static_cast<int>((rect_max.y - rect_min.y) * (rect_max.x - rect_min.x));
            }
        });
    }).wait();
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationPreprocess = end - start;

    int numRendered = 0;
    uint32_t numPoints = gs.getSize();

    std::vector<uint64_t> gaussian_keys_unsorted;  // Adjust size later
    std::vector<uint32_t> gaussian_values_unsorted;  // Adjust size later

    {
        const auto& numTilesTouchedHost = numTilesTouchedBuffer.get_host_access<>();
        for (size_t i = 1; i < numTilesTouchedHost.size(); ++i) {
            numTilesTouchedHost[i] += numTilesTouchedHost[i - 1];
        }
        numRendered = numTilesTouchedHost[numTilesTouchedHost.size() - 1];

        gaussian_keys_unsorted.resize(numRendered * 2);
        gaussian_values_unsorted.resize(numRendered * 2);

               const auto& gaussianBufferHost = pointsBuffer.get_host_access<>();


                      //duplicateWithKeys(numPoints, gaussianBufferHost.get_pointer(), numTilesTouchedHost.get_pointer(),gaussian_keys_unsorted, gaussian_values_unsorted, tileGrid);


                      for (int idx = 0; idx < numPoints; ++idx) {
                          const auto &gaussian = gaussianBufferHost[idx];

                          // Generate no key/value pair for invisible Gaussians
                          if (gaussian.radius > 0) {
                              // Find this Gaussian's offset in buffer for writing keys/values.
                              uint32_t off = (idx == 0) ? 0 : numTilesTouchedHost[idx - 1];
                              glm::ivec2 rect_min, rect_max;

                              getRect(gaussian.screenPos, gaussian.radius, rect_min, rect_max, tileGrid);

                              // For each tile that the bounding rect overlaps, emit a key/value pair.
                              for (int y = rect_min.y; y < rect_max.y; ++y) {
                                  for (int x = rect_min.x; x < rect_max.x; ++x) {
                                      uint64_t key = static_cast<uint64_t>(y) * tileGrid.x + x;
                                      key <<= 32;
                                      key |= *reinterpret_cast<const uint32_t *>(&gaussian.depth);
                                      gaussian_keys_unsorted[off] = key;
                                      gaussian_values_unsorted[off] = static_cast<uint32_t>(idx);
                                      ++off;
                                  }
                              }
                          }
                      }

    }


    // Create a vector of indices for sorting
    std::vector<size_t> indices(numRendered);
    for (size_t i = 0; i < numRendered; ++i) {
        indices[i] = i;
    }
    BinningState binningState{};
    // Ensure the destination vectors have the same size
    binningState.point_list_keys.resize(numRendered);
    binningState.point_list.resize(numRendered);
    binningState.point_list_keys_unsorted = gaussian_keys_unsorted; // Fill with data
    binningState.point_list_unsorted = gaussian_values_unsorted; // Fill with data

    uint32_t numTiles = (imageWidth * imageHeight) / (tileWidth * tileHeight);

    // Sort indices based on keys
    std::sort(indices.begin(), indices.end(),
              [&binningState](size_t a, size_t b) {
                  return binningState.point_list_keys_unsorted[a] < binningState.point_list_keys_unsorted[b];
              });

    // Reorder keys and values based on sorted indices
    for (size_t i = 0; i < numRendered; ++i) {
        binningState.point_list_keys[i] = binningState.point_list_keys_unsorted[indices[i]];
        binningState.point_list[i] = binningState.point_list_unsorted[indices[i]];
    }

    // Initialize imgState.ranges to zero
    ImgState imgState;
    uint32_t num_tiles = tileGrid.x * tileGrid.y;
    imgState.ranges.resize(num_tiles * 2 + 1, std::numeric_limits<uint32_t>::max());

    // Identify tile ranges
    if (numRendered > 0) {
        identifyTileRanges(numRendered, binningState.point_list_keys, imgState.ranges, tileGrid);
    }

    // Print ranges for verification
    for (uint32_t i = 0; i < num_tiles; ++i) {
        uint32_t start = imgState.ranges[i * 2];
        uint32_t end = imgState.ranges[i * 2 + 1];
        if (start != std::numeric_limits<uint32_t>::max()) {
            std::cout << "Tile " << i << ": Start = " << start << ", End = " << end << '\n';
        }
    }

    std::cout << "Num Rendered/NumPoints: " << numRendered << "/" << numPoints << std::endl;


    sycl::buffer<uint32_t, 1> rangesBuffer(imgState.ranges.data(), imgState.ranges.size());
    sycl::range<2> localWorkSize(tileHeight, tileWidth);
    // Compute the global work size ensuring it is a multiple of the local work size
    size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
    sycl::range<2> globalWorkSize(globalHeight, globalWidth);


    start = std::chrono::high_resolution_clock::now();
    queue.submit([&](sycl::handler &h) {

        auto rangesBufferAccess = rangesBuffer.get_access<sycl::access::mode::read>(h);
        auto gaussianBufferAccess = pointsBuffer.get_access<sycl::access::mode::read>(h);

        auto imageAccessor = imageBuffer.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::nd_range<2>(globalWorkSize, localWorkSize), [=](sycl::nd_item<2> item) {


         auto global_id = item.get_global_id(); // Get global indices of the work item
         size_t row = global_id[0];
         size_t col = global_id[1];



         if (row < imageHeight && col < imageWidth) {
             uint32_t tileID = calculateTileID(col, row, imageWidth, tileWidth, tileHeight);
             uint32_t startIdx = rangesBufferAccess[tileID * 2];
             uint32_t endIdx = rangesBufferAccess[tileID * 2 + 1];
             if (startIdx == std::numeric_limits<uint32_t>::max()) {
                 return; // No data for this tile
             }

             // Initialize helper variables
             float T = 1.0f;
             float C[3] = {0};
             for (size_t id = startIdx; id < endIdx; ++id) {
                 const GaussianPoint &point = gaussianBufferAccess[id];
                     // Perform processing on the point and update the image
                     // Example: Set the pixel to a specific value
                     glm::vec2 pos = point.screenPos;
                     // Calculate the exponent term
                     glm::vec2 diff = glm::vec2(col, row) - pos;
                     glm::vec3 c = point.conic;
                     glm::mat2 V(c.x, c.y, c.y, c.z);
                     float power = -0.5f * glm::dot(diff, V * diff);
                     //float power = -0.5f * ((c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy);
                     //double power = -((std::pow(dx, 2) / (2 * std::pow(c.x, 2))) + (std::pow(dy, 2) / (2 * std::pow(c.y, 2))));
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
             imageAccessor[row][col][0] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
             imageAccessor[row][col][1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
             imageAccessor[row][col][2] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
             imageAccessor[row][col][3] = static_cast<uint8_t>(255.0f);

         }
        });
    }).wait();


    std::chrono::duration<double> durationEvaluate = std::chrono::high_resolution_clock::now() - start;
    std::cout << "SYCL RASTERIZE execution time: " << durationEvaluate.count() << "s "
              << "SYCL PREPROCESS execution time: " << durationPreprocess.count() << " seconds" << std::endl;
    auto hostImageAccessor = imageBuffer.get_host_access();
    img = hostImageAccessor.get_pointer();


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
