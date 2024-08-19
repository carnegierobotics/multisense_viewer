//
// Created by magnus on 8/15/24.
//

#include "GaussianModelComponent.h"
#include "tinyply.h"
#include "Viewer/Tools/Utils.h"
#include <glm/ext/quaternion_float.hpp>

namespace VkRender {

    GaussianModelComponent::GaussianModelComponent(std::filesystem::path filePath) {
        m_gaussians = loadFromFile(filePath, 1);
        m_imageComponent = std::make_unique<MeshComponent>(Utils::getModelsPath() / "obj" / "quad.obj");
    }



    GaussianModelComponent::GaussianPoints GaussianModelComponent::loadFromFile(std::filesystem::path path, int downSampleRate) {
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

        std::shared_ptr<tinyply::PlyData> vertices, normals, scales, quats, opacities, colors, harmonics;

        try { vertices = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"}); }
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
        }        // Process vertices
        if (normals) {
            const size_t numVerticesBytes = normals->buffer.size_bytes();
            std::vector<float> vertexBuffer(numVertices * 3);
            std::memcpy(vertexBuffer.data(), normals->buffer.get(), numVerticesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.normals.emplace_back(vertexBuffer[i * 3], vertexBuffer[i * 3 + 1], vertexBuffer[i * 3 + 2]);
            }
        }

        // Process scales
        if (scales) {
            const size_t numScalesBytes = scales->buffer.size_bytes();
            std::vector<float> scaleBuffer(numVertices * 3);
            std::memcpy(scaleBuffer.data(), scales->buffer.get(), numScalesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                float sx = expf(scaleBuffer[i * 3]);
                float sy = expf(scaleBuffer[i * 3 + 1]);
                float sz = expf(scaleBuffer[i * 3 + 2]);

                data.scales.emplace_back(sx, sy, sz);
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
        /*
        if (colors && harmonics) {
            const size_t numColorsBytes = colors->buffer.size_bytes();
            std::vector<float> colorBuffer(numVertices * 3);
            std::memcpy(colorBuffer.data(), colors->buffer.get(), numColorsBytes);

            const size_t numHarmonicsBytes = harmonics->buffer.size_bytes();
            std::vector<float> harmonicsBuffer(numVertices * harmonics_properties.size());
            std::memcpy(harmonicsBuffer.data(), harmonics->buffer.get(), numHarmonicsBytes);

            // Extract DC components
            std::vector<float> features_dc(numVertices * 3);
            for (size_t i = 0; i < numVertices; ++i) {
                features_dc[i * 3 + 0] = colorBuffer[i * 3 + 0];
                features_dc[i * 3 + 1] = colorBuffer[i * 3 + 1];
                features_dc[i * 3 + 2] = colorBuffer[i * 3 + 2];
            }

            // Extract extra features
            std::vector<float> features_extra(numVertices * harmonics_properties.size());
            for (size_t i = 0; i < harmonics_properties.size(); ++i) {
                const size_t offset = i * numVertices;
                for (size_t j = 0; j < numVertices; ++j) {
                    features_extra[j * harmonics_properties.size() + i] = harmonicsBuffer[
                            j * harmonics_properties.size() + i];
                }
            }
            uint32_t max_sh_degree = 3;

            // Reshape and transpose features_extra
            const size_t sh_coeffs = (max_sh_degree + 1) * (max_sh_degree + 1) - 1;
            std::vector<float> reshaped_extra(numVertices * 3 * sh_coeffs);
            for (size_t i = 0; i < numVertices; ++i) {
                for (size_t j = 0; j < sh_coeffs; ++j) {
                    for (size_t k = 0; k < 3; ++k) {
                        reshaped_extra[(i * sh_coeffs + j) * 3 + k] = features_extra[(i * 3 + k) * sh_coeffs + j];
                    }
                }
            }

            // Combine features_dc and reshaped_extra
            data.sphericalHarmonics.resize(numVertices * (3 + sh_coeffs));
            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                for (size_t j = 0; j < 3; ++j) {
                    data.sphericalHarmonics.push_back(features_dc[i * 3 + j]);
                }
                for (size_t j = 0; j < sh_coeffs; ++j) {
                    for (size_t k = 0; k < 3; ++k) {
                        data.sphericalHarmonics.push_back(reshaped_extra[(i * sh_coeffs + j) * 3 + k]);
                    }
                }
            }

            data.shDim = 3 + harmonics_properties.size();
        }
         */

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

            data.shDim = harmonics_properties.size() + 3;
        }
        return data;
    }


}