//
// Created by magnus on 4/28/25.
//

#ifndef QUADRICPOINTCLOUDLOADER_H
#define QUADRICPOINTCLOUDLOADER_H

#include <Viewer/Assets/IAssetLoader.h>
#include <tinyply.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace VkRender {

// Asset holding quadric‐point‐cloud data
struct QuadricCloudAsset : BaseAsset {
    std::vector<glm::vec3> positions;
    std::vector<glm::quat> rotations;
    std::vector<float>     a, b, c;
    std::vector<float>     t_x, t_y;
    std::vector<float>     kernelScale, threshold, beta;

    uint32_t numPoints = 0;

    // Reserve for n quadrics
    void reserve(size_t n) {
        positions.reserve(n);
        rotations.reserve(n);
        a.reserve(n); b.reserve(n); c.reserve(n);
        t_x.reserve(n); t_y.reserve(n);
        kernelScale.reserve(n);
        threshold.reserve(n);
        beta.reserve(n);
    }
};

// Loader for .ply quadric‐point clouds
class QuadricLoader : public IAssetLoader {
public:
    bool canLoad(const std::string &key) const override {
        return hasExtension(key, { ".ply" });
    }

    std::shared_ptr<BaseAsset> load(const std::string &key) override {
        std::filesystem::path path{ key };
        std::ifstream stream(path, std::ios::binary);
        if (!stream.is_open())
            throw std::runtime_error("Failed to open QuadricAsset: " + path.string());

        tinyply::PlyFile ply;
        ply.parse_header(stream);

        // Request all needed properties
        auto vertexData       = ply.request_properties_from_element("vertex", {"x","y","z"});
        auto rotationData     = ply.request_properties_from_element("vertex", {"rot_0","rot_1","rot_2","rot_3"});
        auto aData            = ply.request_properties_from_element("vertex", {"a"});
        auto bData            = ply.request_properties_from_element("vertex", {"b"});
        auto cData            = ply.request_properties_from_element("vertex", {"c"});
        auto t_xData          = ply.request_properties_from_element("vertex", {"t_x"});
        auto t_yData          = ply.request_properties_from_element("vertex", {"t_y"});
        auto kernelScaleData  = ply.request_properties_from_element("vertex", {"kernel_scale"});
        auto thresholdData    = ply.request_properties_from_element("vertex", {"threshold"});
        auto betaData         = ply.request_properties_from_element("vertex", {"beta"});

        // Read the data
        ply.read(stream);

        size_t count = vertexData->count;
        // Validate that each property has the same count
        if (rotationData->count    != count ||
            aData->count           != count ||
            bData->count           != count ||
            cData->count           != count ||
            t_xData->count         != count ||
            t_yData->count         != count ||
            kernelScaleData->count != count ||
            thresholdData->count   != count ||
            betaData->count        != count)
        {
            throw std::runtime_error("Inconsistent vertex count among properties in: " + path.string());
        }

        // Copy raw buffers into vectors
        std::vector<float> posBuf(count * 3);
        std::memcpy(posBuf.data(), vertexData->buffer.get(), count * 3 * sizeof(float));

        std::vector<float> rotBuf(count * 4);
        std::memcpy(rotBuf.data(), rotationData->buffer.get(), count * 4 * sizeof(float));

        std::vector<float> aBuf(count);
        std::memcpy(aBuf.data(), aData->buffer.get(), count * sizeof(float));
        std::vector<float> bBuf(count);
        std::memcpy(bBuf.data(), bData->buffer.get(), count * sizeof(float));
        std::vector<float> cBuf(count);
        std::memcpy(cBuf.data(), cData->buffer.get(), count * sizeof(float));

        std::vector<float> txBuf(count);
        std::memcpy(txBuf.data(), t_xData->buffer.get(), count * sizeof(float));
        std::vector<float> tyBuf(count);
        std::memcpy(tyBuf.data(), t_yData->buffer.get(), count * sizeof(float));

        std::vector<float> ksBuf(count);
        std::memcpy(ksBuf.data(), kernelScaleData->buffer.get(), count * sizeof(float));
        std::vector<float> thrBuf(count);
        std::memcpy(thrBuf.data(), thresholdData->buffer.get(), count * sizeof(float));
        std::vector<float> betaBuf(count);
        std::memcpy(betaBuf.data(), betaData->buffer.get(), count * sizeof(float));

        // Build the asset
        auto asset = std::make_shared<QuadricCloudAsset>();
        asset->reserve(count);

        for (size_t i = 0; i < count; ++i) {
            // Position
            asset->positions.emplace_back(
                posBuf[i*3 + 0],
                posBuf[i*3 + 1],
                posBuf[i*3 + 2]
            );
            // Rotation quaternion (w, x, y, z)
            asset->rotations.emplace_back(
                rotBuf[i*4 + 0],
                rotBuf[i*4 + 1],
                rotBuf[i*4 + 2],
                rotBuf[i*4 + 3]
            );
            // Quadric parameters
            asset->a.push_back(aBuf[i]);
            asset->b.push_back(bBuf[i]);
            asset->c.push_back(cBuf[i]);
            asset->t_x.push_back(txBuf[i]);
            asset->t_y.push_back(tyBuf[i]);
            asset->kernelScale.push_back(ksBuf[i]);
            asset->threshold.push_back(thrBuf[i]);
            asset->beta.push_back(betaBuf[i]);
        }
        asset->numPoints = count;

        return asset;
    }

private:
    // Helper to check file extension
    bool hasExtension(const std::string &s, std::initializer_list<std::string> exts) const {
        auto ext = std::filesystem::path(s).extension().string();
        for (auto &e : exts) if (ext == e) return true;
        return false;
    }
};

}


#endif //QUADRICPOINTCLOUDLOADER_H
