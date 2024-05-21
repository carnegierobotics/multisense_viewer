//
// Created by magnus on 5/17/24.
//

#include "GaussianRenderer.h"
#include "Viewer/Tools/Utils.h"

#include <filesystem>
#include <tinyply.h>

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
        std::cerr << "CPU device not found: " << e.what() << '\n';
        std::cerr << "Falling back to default device selector.\n";
        // Fallback to default device selector
        queue = sycl::queue();
    }
    Log::Logger::getInstance()->info("Selected Device {}",queue.get_device().get_info<sycl::info::device::name>().c_str());

    gs = loadFromFile(100);
    Log::Logger::getInstance()->info("Loaded {} Gaussians", gs.getSize());

    positionBuffer = {gs.positions.data(), sycl::range<1>(gs.getSize())};
    scalesBuffer = {gs.scales.data(), sycl::range<1>(gs.getSize())};
    quaternionBuffer = {gs.quats.data(), sycl::range<1>(gs.getSize())};
    opacityBuffer = {gs.opacities.data(), sycl::range<1>(gs.getSize())};

    // Create a buffer to store the resulting 2D covariance vectors
    covariance2DBuffer = {sycl::range<1>(gs.getSize())};
    conicBuffer = {sycl::range<1>(gs.getSize())};
    screenPosBuffer = sycl::buffer<glm::vec3, 1>{sycl::range<1>(gs.getSize())};
    covarianceBuffer = {sycl::range<1>(gs.getSize())};

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

    simpleRasterizer(camera);
}

glm::mat3 computeCov3D(const glm::vec3 &scale, const glm::quat &q) {
    glm::mat3 S(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;

    glm::mat3 R = glm::mat3_cast(q);

    glm::mat3 M = S * R;
    glm::mat3 St = glm::transpose(S);
    glm::mat3 Rt = glm::transpose(R);
    glm::mat3 Sigma = R * S * St * Rt;

    //glm::mat3 Sigma = M * transpose(M);
    return Sigma;
}

glm::vec3 computeCov2D(const glm::vec4 &pView,
                       const glm::mat3 &cov3D, const glm::mat4 &viewMat, const GaussianRenderer::CameraParams &camera) {
    glm::vec4 t = pView;
    const float limx = 1.3f * camera.tanFovX;
    const float limy = 1.3f * camera.tanFovY;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
    t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

    float l = glm::length(glm::vec3(t));
    glm::mat3 J = glm::mat3(camera.focalX / t.z, 0.0f, t.x / l,
                            0.0f, camera.focalY / t.z, t.y / l,
                            -(camera.focalX * t.x) / (t.z * t.z), -(camera.focalY * t.y) / (t.z * t.z), t.z / l);

    auto W = glm::mat3(viewMat);
    glm::mat3 T = J * W;
    glm::mat3 cov = J * W * cov3D * glm::transpose(W) * glm::transpose(J);
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    //cov = glm::transpose(cov);

    return {cov[0][0], cov[0][1], cov[1][1]};

}


void GaussianRenderer::simpleRasterizer(const VkRender::Camera &camera) {

    auto params = getHtanfovxyFocal(camera.m_Fov, camera.m_height, camera.m_width);

    glm::mat4 viewMatrix = camera.matrices.view;
    glm::mat4 projectionMatrix = camera.matrices.perspective;

    /*
    float aspect =  static_cast<float>(camera.m_width) / static_cast<float>(camera.m_height);
    float focalLength = 1.0f / tanf(glm::radians(camera.m_Fov) * 0.5f);
    float x = focalLength / aspect;
    float y = focalLength;
    float A = camera.m_Zfar / (camera.m_Zfar - camera.m_Znear);
    float B = (-camera.m_Zfar * camera.m_Znear) / (camera.m_Zfar - camera.m_Znear);
    projectionMatrix = glm::mat4(
            x, 0.0f, 0.0f, 0.0f,
            0.0f, y, 0.0f, 0.0f,
            0.0f, 0.0f, A, 1.0f,
            0.0f, 0.0f, B, 0.0f
    );
    */

    const size_t imageWidth = camera.m_width;
    const size_t imageHeight = camera.m_height;

    queue.submit([&](sycl::handler &h) {
        auto scales = scalesBuffer.get_access<sycl::access::mode::read>(h);
        auto quaternions = quaternionBuffer.get_access<sycl::access::mode::read>(h);
        auto positions = positionBuffer.get_access<sycl::access::mode::read>(h);

        auto covariances = covarianceBuffer.get_access<sycl::access::mode::write>(h);
        auto covariances2D = covariance2DBuffer.get_access<sycl::access::mode::write>(h);
        auto conic = conicBuffer.get_access<sycl::access::mode::write>(h);
        auto screenPos = screenPosBuffer.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(gs.getSize()), [=](sycl::id<1> idx) {
            glm::vec3 scale = scales[idx];
            glm::quat q = quaternions[idx];
            glm::vec3 position = positions[idx];
            covariances[idx] = computeCov3D(scale, q);
            glm::vec4 posView = viewMatrix * glm::vec4(position, 1.0f);

            //covariances2D[idx] = computeCov2D(posView, focalX, focalY, tanX, tanY, covariances[idx], viewMatrix);
            covariances2D[idx] = computeCov2D(posView, covariances[idx], viewMatrix, params);

            // Invert covariance (EWA)
            const auto &cov2D = covariances2D[idx];
            float determinant = cov2D.x * cov2D.z - (cov2D.y * cov2D.y);
            if (determinant == 0) return;

            float invDeterminant = 1 / determinant;
            conic[idx] = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant, cov2D.x * invDeterminant);

            glm::vec4 posClip = projectionMatrix * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;

            // Transform to framebuffer coordinates
            float framebufferX = (posNDC.x * 0.5f + 0.5f) * imageWidth;
            float framebufferY = (posNDC.y * 0.5f + 0.5f) * imageHeight; // Flip (Vulkan framebuffer coordinates)
            screenPos[idx] = glm::vec3(framebufferX, framebufferY, posNDC.z);
        });
    }).wait();

    sycl::range<2> localWorkSize(16, 16);
    // Compute the global work size ensuring it is a multiple of the local work size
    size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
    sycl::range<2> globalWorkSize(globalHeight, globalWidth);
    queue.submit([&](sycl::handler &h) {
        auto conic = conicBuffer.get_access<sycl::access::mode::read>(h);
        auto screenPosGaussian = screenPosBuffer.get_access<sycl::access::mode::read>(h);
        auto imageAccessor = imageBuffer.get_access<sycl::access::mode::write>(h);
        auto pngImageAccessor = pngImageBuffer.get_access<sycl::access::mode::read>(h);
        auto positions = positionBuffer.get_access<sycl::access::mode::read>(h);

        h.parallel_for(sycl::nd_range<2>(globalWorkSize, localWorkSize), [=](sycl::nd_item<2> item) {
            auto global_id = item.get_global_id(); // Get global indices of the work item
            size_t row = global_id[0];
            size_t col = global_id[1];

            if (row < imageHeight && col < imageWidth) {
                float maxInfluence = 0.0f;
                glm::vec3 rgb(0.0f);
                for (size_t id = 0; id < screenPosGaussian.size(); ++id) {
                    glm::vec3 pos = screenPosGaussian[id];
                    glm::vec3 c = conic[id];

                    float dx = col - pos.x;
                    float dy = row - pos.y;
                    float power = -0.5f * ((c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy);
                    float influence = std::exp(power);
                    if (power > 0.0f)
                        continue;

                    if (influence > maxInfluence) {
                        maxInfluence = influence;
                        float finalColor = influence * 255.0f;
                        switch (id) {
                            case 1:
                                rgb.x = finalColor;
                                break;
                            case 2:
                                rgb.y = finalColor;
                                break;
                            case 3:
                                rgb.z = finalColor;
                                break;
                            default:
                                rgb = glm::vec3(finalColor);
                        }
                    }


                }
                size_t rowFlipped = imageHeight - row - 1;

                imageAccessor[rowFlipped][col][0] = static_cast<uint8_t>(rgb.x);
                imageAccessor[rowFlipped][col][1] = static_cast<uint8_t>(rgb.y);
                imageAccessor[rowFlipped][col][2] = static_cast<uint8_t>(rgb.z);

                //imageAccessor[row][col][0] = pngImageAccessor[row][col][0];
                //imageAccessor[row][col][1] = pngImageAccessor[row][col][1];
                //imageAccessor[row][col][2] = pngImageAccessor[row][col][2];

                imageAccessor[row][col][3] = 255; // Alpha channel remains constant
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
    data.scales.emplace_back(scale);
    //data.scales.emplace_back(scale);
    //data.scales.emplace_back(scale);
    //data.scales.emplace_back(scale);
    data.scales.emplace_back(scale * glm::vec3(5.0f, 1.0f, 1.0f));
    data.scales.emplace_back(scale * glm::vec3(1.0f, 5.0f, 1.0f));
    data.scales.emplace_back(scale * glm::vec3(1.0f, 1.0f, 5.0f));

    for (int i = 0; i < 4; ++i) {
        data.quats.emplace_back(unitQuat);
        data.opacities.emplace_back(1.0f);
    }

    return data;

}

GaussianRenderer::GaussianPoints GaussianRenderer::loadFromFile(int downSampleRate) {
    GaussianPoints data;
    auto plyFilePath = std::filesystem::path("/home/magnus/phd/SuGaR/output/refined_ply/0000/3dgs.ply");

// Open the PLY file
    std::ifstream ss(plyFilePath, std::ios::binary);
    if (!ss.is_open()) {
        throw std::runtime_error("Failed to open PLY file.");
    }

    tinyply::PlyFile file;
    file.parse_header(ss);

    std::shared_ptr<tinyply::PlyData> vertices, scales, quats, opacities;

    try { vertices = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"}); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { quats = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { opacities = file.request_properties_from_element("vertex", {"opacity"}); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


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
            data.scales.emplace_back(scaleBuffer[i * 3], scaleBuffer[i * 3 + 1], scaleBuffer[i * 3 + 2]);
        }
    }

    // Process quats
    if (quats) {
        const size_t numQuatsBytes = quats->buffer.size_bytes();
        std::vector<float> quatBuffer(numVertices * 4);
        std::memcpy(quatBuffer.data(), quats->buffer.get(), numQuatsBytes);

        for (size_t i = 0; i < numVertices; i += downSampleRate) {
            data.quats.emplace_back(quatBuffer[i * 4], quatBuffer[i * 4 + 1], quatBuffer[i * 4 + 2], quatBuffer[i * 4 + 3]);
        }
    }

    // Process opacities
    if (opacities) {
        const size_t numOpacitiesBytes = opacities->buffer.size_bytes();
        std::vector<float> opacityBuffer(numVertices);
        std::memcpy(opacityBuffer.data(), opacities->buffer.get(), numOpacitiesBytes);

        for (size_t i = 0; i < numVertices; i += downSampleRate) {
            data.opacities.push_back(opacityBuffer[i]);
        }
    }

    return data;
}
