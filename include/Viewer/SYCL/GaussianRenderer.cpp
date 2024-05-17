//
// Created by magnus on 5/17/24.
//

#include "GaussianRenderer.h"
#include "Viewer/Tools/Utils.h"

#include <sycl/sycl.hpp>
#include <filesystem>


glm::mat3 computeCov3D(const glm::vec3& scale, const glm::quat& q) {
    glm::mat3 S(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;

    glm::mat3 R = glm::mat3_cast(q);

    glm::mat3 M = S * R;
    glm::mat3 Sigma = transpose(M) * M;
    return Sigma;
}

glm::vec3 computeCov2D(const glm::vec4& mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const glm::mat3& cov3D, const glm::mat4& viewmatrix) {
    glm::vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
    t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
            focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
            0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0.0f, 0.0f, 0.0f
    );

    auto W = glm::mat3(viewmatrix);
    glm::mat3 T = W * J;

    glm::mat3 cov = T * cov3D * transpose(T);
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;

    return {cov[0][0], cov[0][1], cov[1][1]};
}


GaussianRenderer::GaussianRenderer(const VkRender::Camera &camera) {
    simpleRasterizer(camera);
    /*
    std::filesystem::path path = Utils::getTexturePath() / "rover.png";

    uint8_t* image = stbi_load(path.string().c_str(), &m_width, &m_height, &m_channels, STBI_rgb);
    if (!image) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return;
    }

    m_outputImage.resize(m_width * m_height * m_channels);

    sycl::queue queue;
    sycl::buffer<uint8_t, 1> imageBuffer{image, sycl::range<1>(m_width * m_height * m_channels)};
    sycl::buffer<uint8_t, 1> outputBuffer{m_outputImage.data(), sycl::range<1>(m_width * m_height * m_channels)};

    try {
        queue.submit([&](sycl::handler& h) {
            auto in = imageBuffer.get_access<sycl::access::mode::read>(h);
            auto out = outputBuffer.get_access<sycl::access::mode::write>(h);

            int width = m_width;
            int height = m_height;
            int channels = m_channels;

            int Gx[3][3] = {
                    {-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}
            };

            int Gy[3][3] = {
                    {-1, -2, -1},
                    {0,  0,  0},
                    {1,  2,  1}
            };

            h.parallel_for(sycl::range<2>(width, height), [=](sycl::id<2> idx) {
                uint32_t x = idx[0];
                uint32_t y = idx[1];



                auto getPixel = [&](int x, int y, int channel) -> int {
                    if (x < 0 || x >= width || y < 0 || y >= height) {
                        return 0;
                    }
                    return in[((y * width + x) * channels) + channel];
                };

                for (int c = 0; c < 3; c++) {  // Process each color channel separately
                    int sumX = 0;
                    int sumY = 0;
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            sumX += Gx[i + 1][j + 1] * getPixel(x + j, y + i, c);
                            sumY += Gy[i + 1][j + 1] * getPixel(x + j, y + i, c);
                        }
                    }

                    int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
                    if (magnitude > 255) magnitude = 255;
                    if (magnitude < 0) magnitude = 0;

                    uint32_t index = ((y * 1024 + x) * channels) + c;
                    out[index] = magnitude;
                }

            });

        }).wait();
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        stbi_image_free(image);
        return;
    }

    auto hostBuffer = outputBuffer.get_host_access();

    stbi_image_free(image);

    // Optionally, save the output image using stb_image_write
    if (!stbi_write_png("../output.png", m_width, m_height, m_channels, hostBuffer.get_pointer(), m_width * m_channels)) {
        std::cerr << "Error saving the output image" << std::endl;
    }
    */
}

void GaussianRenderer::simpleRasterizer(const VkRender::Camera& camera){
    auto gs = loadNaive();

    std::vector<float> focalParams = getHtanfovxyFocal(camera.m_Fov, camera.m_height, camera.m_width);

    float focalX = focalParams[2];
    float focalY = focalParams[3];
    float tanX = focalParams[0];
    float tanY = focalParams[1];

    sycl::queue queue;
    sycl::buffer<glm::vec3, 1> positionBuffer{gs.positions.data(), sycl::range<1>(gs.getSize())};
    sycl::buffer<glm::vec3, 1> scalesBuffer{gs.scales.data(), sycl::range<1>(gs.getSize())};
    sycl::buffer<glm::quat, 1> quaternionBuffer{gs.quats.data(), sycl::range<1>(gs.getSize())};
    sycl::buffer<float, 1> opacityBuffer{gs.opacities.data(), sycl::range<1>(gs.getSize())};

    // Create a buffer to store the resulting covariance matrices
    std::vector<glm::mat3> covarianceMatrices(gs.getSize());
    sycl::buffer<glm::mat3, 1> covarianceBuffer{covarianceMatrices.data(), sycl::range<1>(gs.getSize())};
    // Create a buffer to store the resulting 2D covariance vectors
    std::vector<glm::vec3> covariance2D(gs.getSize());
    sycl::buffer<glm::vec3, 1> covariance2DBuffer{covariance2D.data(), sycl::range<1>(gs.getSize())};

    // Parameters for computeCov2D
    glm::mat4 viewMatrix = camera.matrices.view;
    queue.submit([&](sycl::handler& h) {
        auto scales = scalesBuffer.get_access<sycl::access::mode::read>(h);
        auto quaternions = quaternionBuffer.get_access<sycl::access::mode::read>(h);
        auto positions = positionBuffer.get_access<sycl::access::mode::read>(h);

        auto covariances = covarianceBuffer.get_access<sycl::access::mode::write>(h);
        auto covariances2D = covariance2DBuffer.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(gs.getSize()), [=](sycl::id<1> idx) {
            glm::vec3 scale = scales[idx];
            glm::quat q = quaternions[idx];
            glm::vec3 position = positions[idx];


            covariances[idx] = computeCov3D(scale, q);
            glm::vec4 mean_view = glm::vec4(position, 1.0f);

            covariances2D[idx] = computeCov2D(mean_view, focalX, focalY, tanX, tanY, covariances[idx], viewMatrix);

        });
    }).wait();

    // Copy data from SYCL buffer to host vector
    auto covHost = covarianceBuffer.get_host_access();
    std::vector<glm::mat3> covariances(gs.getSize());
    std::copy(covHost.get_pointer(), covHost.get_pointer() + gs.getSize(), covariances.begin());
    // The covariance matrices are now copied back to the host and stored in covariances
    for (const auto& cov : covariances) {
        std::cout << "Covariance Matrix:\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << cov[i][0] << " " << cov[i][1] << " " << cov[i][2] << "\n";
        }
    }
    // Copy data from SYCL buffer to host vector
    auto cov2DHost = covariance2DBuffer.get_host_access();
    std::vector<glm::vec3> covariances2D(gs.getSize());
    std::copy(cov2DHost.get_pointer(), cov2DHost.get_pointer() + gs.getSize(), covariances2D.begin());
    // The covariance matrices are now copied back to the host and stored in covariances
    std::cout << "Covariance 2D:\n";
    for (const auto& cov : covariances2D) {
        std::cout << cov.x << " " << cov.y << " " << cov.z << "\n";
    }
}

GaussianRenderer::GaussianPoints GaussianRenderer::loadNaive() {
    GaussianRenderer::GaussianPoints data;
    auto unitQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    auto scale = glm::vec3(0.05f, 0.05f, 0.05f);
    auto pos = glm::vec3(0.0f, 0.0f, 0.0f);

    data.positions.emplace_back(pos);
    data.positions.emplace_back(pos + glm::vec3(1.0f, 0.0f, 0.0f));
    data.positions.emplace_back(pos + glm::vec3(0.0f, 1.0f, 0.0f));
    data.positions.emplace_back(pos + glm::vec3(0.0f, 0.0f, 1.0f));
    data.scales.emplace_back(scale);
    data.scales.emplace_back(scale * glm::vec3(5.0f, 1.0f, 1.0f));
    data.scales.emplace_back(scale * glm::vec3(1.0f, 5.0f, 1.0f));
    data.scales.emplace_back(scale * glm::vec3(1.0f, 1.0f, 5.0f));

    for (int i = 0; i < 4; ++i){
        data.quats.emplace_back(unitQuat);
        data.opacities.emplace_back(1.0f);
    }

    return data;

}
