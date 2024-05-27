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
    loadedPly = false;
    setupBuffers(camera);

    simpleRasterizer(camera, false);
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

    int width, height, channels;
    std::filesystem::path path = Utils::getTexturePath() / "moon.png";
    image = stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!image) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return;
    }

    //sycl::buffer<uint8_t, 1> pngImageBuffer{image, sycl::range<1>(width * height * channels)};
    pngImageBuffer = {image, sycl::range<3>(height, width, 4)};
    imageBuffer = {sycl::range<3>(camera.m_height, camera.m_width, 4)};
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

void GaussianRenderer::simpleRasterizer(const VkRender::Camera &camera, bool debug) {

    auto params = getHtanfovxyFocal(camera.m_Fov, camera.m_height, camera.m_width);

    glm::mat4 viewMatrix = camera.matrices.view;
    glm::mat4 projectionMatrix = camera.matrices.perspective;
    glm::vec3 camPos = camera.pose.pos;

    // Flip the second row of the projection matrix
    projectionMatrix[1] = -projectionMatrix[1];

    const size_t imageWidth = camera.m_width;
    const size_t imageHeight = camera.m_height;

    std::vector<float> depths(gs.getSize());
    std::vector<int> indices(gs.getSize());

    sycl::buffer<float, 1> depthsBuffer(depths.data(), sycl::range<1>(gs.getSize()));
    sycl::buffer<int, 1> indicesBuffer(indices.data(), sycl::range<1>(gs.getSize()));

    /*
    if (debug) {
        size_t i = 1;
        glm::vec3 position = gs.positions[i];
        glm::vec4 posView = viewMatrix * glm::vec4(position, 1.0f);
        glm::vec3 scale = gs.scales[i];
        glm::quat q = gs.quats[i];
        glm::mat3 covariances = computeCov3D(scale, q);

        const auto &cov2D = computeCov2D(posView, covariances, viewMatrix, params, debug);
        float determinant = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
        if (determinant != 0) {
            float invDeterminant = 1.0f / determinant;
            auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant, cov2D.x * invDeterminant);

            sycl::ext::oneapi::experimental::printf("Conic\n");
            sycl::ext::oneapi::experimental::printf("%f %f %f, inv_det: %f\n", conic.x, conic.y, conic.z,
                                                    invDeterminant);
        }

    }
 */
    queue.submit([&](sycl::handler &h) {
        auto scales = scalesBuffer.get_access<sycl::access::mode::read>(h);
        auto quaternions = quaternionBuffer.get_access<sycl::access::mode::read>(h);
        auto positions = positionBuffer.get_access<sycl::access::mode::read>(h);
        auto shs = sphericalHarmonicsBuffer.get_access<sycl::access::mode::read>(h);

        auto covariances = covarianceBuffer.get_access<sycl::access::mode::write>(h);
        auto covariances2D = covariance2DBuffer.get_access<sycl::access::mode::write>(h);
        auto conic = conicBuffer.get_access<sycl::access::mode::write>(h);
        auto screenPos = screenPosBuffer.get_access<sycl::access::mode::write>(h);
        auto colorOutput = colorOutputBuffer.get_access<sycl::access::mode::write>(h);

        auto depthsAccess = depthsBuffer.get_access<sycl::access::mode::write>(h);
        auto indicesAccess = indicesBuffer.get_access<sycl::access::mode::write>(h);

        uint32_t shDim = gs.getShDim();

        h.parallel_for(sycl::range<1>(gs.getSize()), [=](sycl::id<1> idx) {
            glm::vec3 scale = scales[idx];
            glm::quat q = quaternions[idx];
            glm::vec3 position = positions[idx];
            glm::vec4 posView = viewMatrix * glm::vec4(position, 1.0f);
            glm::vec4 posClip = projectionMatrix * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;

            glm::vec3 threshold(1.3f);
            // Manually compute absolute values
            glm::vec3 pos_screen_abs = glm::vec3(
                    std::abs(posNDC.x),
                    std::abs(posNDC.y),
                    std::abs(posNDC.z)
            );
            if (glm::any(glm::greaterThan(pos_screen_abs, threshold))) {
                screenPos[idx] = glm::vec4(-100.0f, -100.0f, -100.0f, 1.0f);
                //sycl::ext::oneapi::experimental::printf("Culled: x = %f, y = %f, z = %f\n", position.x, position.y, position.z);
                return;
            }
            covariances[idx] = computeCov3D(scale, q);
            covariances2D[idx] = computeCov2D(posView, covariances[idx], viewMatrix, params, false);
            // Invert covariance (EWA)
            const auto &cov2D = covariances2D[idx];
            float determinant = cov2D.x * cov2D.z - (cov2D.y * cov2D.y);
            if (determinant != 0) {
                float invDeterminant = 1 / determinant;
                conic[idx] = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant, cov2D.x * invDeterminant);
                screenPos[idx] = posNDC;
                // Calculate spherical harmonics to color:
                glm::vec3 dir = glm::normalize(position - camPos);
                glm::vec3 color =
                        SH_C0 * glm::vec3(shs[idx * shDim + 0], shs[(idx * shDim) + 1], shs[(idx * shDim) + 2]);

                color += 0.5f;
                colorOutput[idx] = color;
                depthsAccess[idx] = posNDC.z; // Depth
                indicesAccess[idx] = idx; // Original index
            }
        });
    }).wait();


    /*

    {
        auto depthsHost = depthsBuffer.get_host_access();
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return depthsHost[a] > depthsHost[b];
        });
    }
    indicesBuffer = sycl::buffer<int, 1>(indices.data(), sycl::range<1>(gs.getSize()));

    sycl::buffer<glm::vec3, 1> sortedScreenPositionsBuffer(gs.getSize());
    sycl::buffer<glm::vec3, 1> sortedColorsBuffer(gs.getSize());
    sycl::buffer<glm::vec3, 1> sortedConicsBuffer(gs.getSize());
    sycl::buffer<float, 1> sortedOpacitiesBuffer(gs.getSize());
    sycl::buffer<float, 1> sortedSphericalHarmonicsBuffer(gs.getSize() * gs.getShDim());
    const uint32_t shDim = gs.getShDim();
    const uint32_t numGaussians = gs.getSize();
    queue.submit([&](sycl::handler &h) {
        auto positions = screenPosBuffer.get_access<sycl::access::mode::read>(h);
        auto color = colorOutputBuffer.get_access<sycl::access::mode::read>(h);
        auto conic = conicBuffer.get_access<sycl::access::mode::read>(h);
        auto opacities = opacityBuffer.get_access<sycl::access::mode::read>(h);
        auto sphericalHarmonics = sphericalHarmonicsBuffer.get_access<sycl::access::mode::read>(h);
        auto indicesAccess = indicesBuffer.get_access<sycl::access::mode::read>(h);

        auto sortedPositionsAccess = sortedScreenPositionsBuffer.get_access<sycl::access::mode::write>(h);
        auto sortedColorsAccess = sortedColorsBuffer.get_access<sycl::access::mode::write>(h);
        auto sortedConicsAccess = sortedConicsBuffer.get_access<sycl::access::mode::write>(h);
        auto sortedOpacitiesAccess = sortedOpacitiesBuffer.get_access<sycl::access::mode::write>(h);
        auto sortedSphericalHarmonicsAccess = sortedSphericalHarmonicsBuffer.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(numGaussians), [=](sycl::id<1> idx) {
            int sortedIdx = indicesAccess[idx];
            sortedPositionsAccess[idx] = positions[sortedIdx];
            sortedColorsAccess[idx] = color[sortedIdx];
            sortedConicsAccess[idx] = conic[sortedIdx];
            sortedOpacitiesAccess[idx] = opacities[sortedIdx];
            for (size_t j = 0; j < shDim; ++j) {
                sortedSphericalHarmonicsAccess[idx * shDim + j] = sphericalHarmonics[sortedIdx * shDim + j];
            }
        });
    }).wait();
    */


    sycl::range<2> localWorkSize(16, 16);
    // Compute the global work size ensuring it is a multiple of the local work size
    size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
    sycl::range<2> globalWorkSize(globalHeight, globalWidth);
    queue.submit([&](sycl::handler &h) {
/*
        auto screenPosGaussian = sortedScreenPositionsBuffer.get_access<sycl::access::mode::read>(h);
        auto colors = sortedColorsBuffer.get_access<sycl::access::mode::read>(h);
        auto conic = sortedConicsBuffer.get_access<sycl::access::mode::read>(h);
        auto opacities = sortedOpacitiesBuffer.get_access<sycl::access::mode::read>(h);

*/
        auto screenPosGaussian = screenPosBuffer.get_access<sycl::access::mode::read>(h);
        auto colors = colorOutputBuffer.get_access<sycl::access::mode::read>(h);
        auto conic = conicBuffer.get_access<sycl::access::mode::read>(h);
        auto opacities = opacityBuffer.get_access<sycl::access::mode::read>(h);

        auto imageAccessor = imageBuffer.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::nd_range<2>(globalWorkSize, localWorkSize), [=](sycl::nd_item<2> item) {
            auto global_id = item.get_global_id(); // Get global indices of the work item
            size_t row = global_id[0];
            size_t col = global_id[1];

            if (row < imageHeight && col < imageWidth) {
                // Initialize helper variables
                float T = 1.0f;
                float C[3] = {0};
                for (int id = screenPosGaussian.size() - 1; id >= 0; --id) { // Iterate from back to front
                    glm::vec3 pos = screenPosGaussian[id];

                    float pixPosX = ((pos.x + 1.0f) * imageWidth - 1.0f)  * 0.5f;
                    float pixPosY = ((pos.y + 1.0f) * imageHeight - 1.0f) * 0.5f;
                    // Calculate the exponent term
                    glm::vec2 diff = glm::vec2(col, row) - glm::vec2(pixPosX, pixPosY);
                    glm::vec3 c = conic[id];
                    glm::mat2 V(c.x, c.y, c.y, c.z);
                    float power = -0.5f * glm::dot(diff, V * diff);
                    //float power = -0.5f * ((c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy);
                    //double power = -((std::pow(dx, 2) / (2 * std::pow(c.x, 2))) + (std::pow(dy, 2) / (2 * std::pow(c.y, 2))));
                    if (power > 0.0f) {
                        continue;
                    }
                    float alpha = std::min(0.99f, opacities[id] * expf(power));

                    if (alpha < 1.0f / 255.0f)
                        continue;

                    float test_T = T * (1 - alpha);
                    if (test_T < 0.0001f) {
                        continue;
                    }
                    // Eq. (3) from 3D Gaussian splatting paper.
                    for (int ch = 0; ch < 3; ch++) {
                        C[ch] += colors[id][ch] * alpha * T;
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
    auto hostImageAccessor = imageBuffer.get_host_access();
    img = hostImageAccessor.get_pointer();


    /*
    // Copy data from SYCL buffer to host vector
    auto cov2DHost = covariance2DBuffer.get_host_access();
    auto screenPosHost = screenPosBuffer.get_host_access();
    auto conicHost = conicBuffer.get_host_access();
    std::vector<glm::vec3> covariances2D(gs.getSize());
    std::vector<glm::vec3> screenPos(gs.getSize());
    std::vector<glm::vec3> conic(gs.getSize());
    std::copy(cov2DHost.get_pointer(), cov2DHost.get_pointer() + gs.getSize(), covariances2D.begin());
    std::copy(screenPosHost.get_pointer(), screenPosHost.get_pointer() + gs.getSize(), screenPos.begin());
    std::copy(conicHost.get_pointer(), conicHost.get_pointer() + gs.getSize(), conic.begin());
    // The covariance matrices are now copied back to the host and stored in covariances
    std::cout << "Data 2D:\n";
    for (size_t i = 0; i < covariances2D.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6); // Set precision to 3 decimal places

        std::cout << "Pos3D (World):       (" << std::setw(8) << gs.positions[i].x << ", "
                  << std::setw(8) << gs.positions[i].y << ", "
                  << std::setw(8) << gs.positions[i].z << ")\n";

        std::cout << "Pos2D (Framebuffer): (" << std::setw(8) << screenPosHost[i].x << ", "
                  << std::setw(8) << screenPosHost[i].y << ", "
                  << std::setw(8) << screenPosHost[i].z << ")\n";

        std::cout << "Cov2D (Screen):      (" << std::setw(8) << covariances2D[i].x << ", "
                  << std::setw(8) << covariances2D[i].y << ", "
                  << std::setw(8) << covariances2D[i].z << ")\n";

        std::cout << "Conic (2D):      (" << std::setw(8) << conic[i].x << ", "
                  << std::setw(8) << conic[i].y << ", "
                  << std::setw(8) << conic[i].z << ")\n\n";
    }

    // Copy data from SYCL buffer to host vector
    auto covHost = covarianceBuffer.get_host_access();
    std::vector<glm::mat3> covariances(gs.getSize());
    std::copy(covHost.get_pointer(), covHost.get_pointer() + gs.getSize(), covariances.begin());
    // The covariance matrices are now copied back to the host and stored in covariances
    for (const auto &cov: covariances) {
        std::cout << "Covariance 3D Matrix:\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << cov[i][0] << " " << cov[i][1] << " " << cov[i][2] << "\n";
        }
        std::cout << "\n";
    }
    */

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
            data.scales.emplace_back(expf(scaleBuffer[i * 3]), expf(scaleBuffer[i * 3 + 1]), expf(scaleBuffer[i * 3 + 2]));
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
