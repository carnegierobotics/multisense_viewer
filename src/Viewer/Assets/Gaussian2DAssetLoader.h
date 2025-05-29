#ifndef TWODGPOINTCLOUDLOADER_H
#define TWODGPOINTCLOUDLOADER_H

#define GLM_ENABLE_EXPERIMENTAL

#include <Viewer/Assets/IAssetLoader.h>
#include <tinyply.h>
#include <glm/gtc/quaternion.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <random>

namespace VkRender {
    //--------------------------------------------------------------------------
    // Asset holding 2‑D Gaussian splat data
    //--------------------------------------------------------------------------
    struct Gaussian2DAsset : BaseAsset {
        std::vector<glm::vec3> positions; // mean µ
        std::vector<glm::quat> rotations; // orientation of the tangent plane (w, x, y, z)
        std::vector<float> scale_x; // σ_x
        std::vector<float> scale_y; // σ_y
        std::vector<glm::vec3> colors; // zero‑order SH (linear RGB)
        std::vector<float> opacity; // α ∈ [0,1]
        std::vector<std::vector<float> > shCoeffs; // higher-order SH (per-point)

        uint32_t numPoints = 0;

        void reserve(size_t n) {
            positions.reserve(n);
            rotations.reserve(n);
            scale_x.reserve(n);
            scale_y.reserve(n);
            colors.reserve(n);
            opacity.reserve(n);
            shCoeffs.reserve(n);
        }
    };

    //--------------------------------------------------------------------------
    // Loader for .ply 2‑D Gaussian splats
    //--------------------------------------------------------------------------
    class Gaussian2DAssetLoader : public IAssetLoader {
    public:
        bool canLoad(const std::string &key) const override {
            return hasExtension(key, {".ply"});
        }

        std::type_index assetType() const override {
            return typeid(Gaussian2DAsset);
        }

        std::shared_ptr<BaseAsset> load(const std::filesystem::path &key) override {
            using tinyply::PlyData;
            std::ifstream stream(key, std::ios::binary);
            if (!stream.is_open())
                throw std::runtime_error("Failed to open Gaussian2DAsset: " + key.string());

            tinyply::PlyFile ply;
            ply.parse_header(stream);

            // Mandatory
            auto posData = ply.request_properties_from_element("vertex", {"x", "y", "z"});
            // Optional unary properties
            auto rotData = tryProperties(ply, "vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});
            auto sxData = tryProperties(ply, "vertex", {"scale_0"});
            auto syData = tryProperties(ply, "vertex", {"scale_1"});
            auto colData = tryProperties(ply, "vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
            auto opaData = tryProperties(ply, "vertex", {"opacity"});


            ply.read(stream);

            const size_t count = posData->count;
            validateCount(rotData, count, "rot_*", key);
            validateCount(sxData, count, "scale_0", key);
            validateCount(syData, count, "scale_1", key);
            validateCount(colData, count, "f_dc_*", key);
            validateCount(opaData, count, "opacity", key);

            // Copy buffers
            std::vector<float> posBuf(count * 3);
            std::memcpy(posBuf.data(), posData->buffer.get(), posBuf.size() * sizeof(float));

            auto copyOrDefault = [&](std::shared_ptr<PlyData> &src, float def) {
                std::vector<float> dst(count, def);
                if (src) std::memcpy(dst.data(), src->buffer.get(), count * sizeof(float));
                return dst;
            };
            auto copyOrDefaultVec4 = [&](std::shared_ptr<PlyData> &src, glm::quat def) {
                std::vector<float> dst(count * 4);
                if (src) std::memcpy(dst.data(), src->buffer.get(), dst.size() * sizeof(float));
                else
                    for (size_t i = 0; i < count; ++i) {
                        dst[i * 4 + 0] = def.w;
                        dst[i * 4 + 1] = def.x;
                        dst[i * 4 + 2] = def.y;
                        dst[i * 4 + 3] = def.z;
                    }
                return dst;
            };

            std::vector<float> rotBuf = copyOrDefaultVec4(rotData, glm::quat{1, 0, 0, 0});
            std::vector<float> sxBuf = copyOrDefault(sxData, std::log(0.3f));
            std::vector<float> syBuf = copyOrDefault(syData, std::log(0.3f));
            std::vector<float> colBuf = [&]() {
                std::vector<float> d(count * 3, 1.f);
                if (colData) std::memcpy(d.data(), colData->buffer.get(), d.size() * sizeof(float));
                return d;
            }();
            std::vector<float> opaBuf = copyOrDefault(opaData, 1.f);


            auto asset = std::make_shared<Gaussian2DAsset>();
            asset->reserve(count);

            for (size_t i = 0; i < count; ++i) {
                asset->positions.emplace_back(
                    posBuf[i * 3 + 0], posBuf[i * 3 + 1], posBuf[i * 3 + 2]
                );
                asset->rotations.emplace_back(
                    glm::normalize(
                        glm::quat(rotBuf[i * 4 + 0], rotBuf[i * 4 + 1], rotBuf[i * 4 + 2], rotBuf[i * 4 + 3]))
                );
                asset->scale_x.push_back(std::exp(sxBuf[i]));
                asset->scale_y.push_back(std::exp(syBuf[i]));
                asset->colors.emplace_back(
                    colBuf[i * 3 + 0], colBuf[i * 3 + 1], colBuf[i * 3 + 2]
                );
                float val = opaBuf[i];
                float alpha = 1.0f / (1.0f + std::exp(-val));
                asset->opacity.push_back(alpha);
            }
            asset->numPoints = uint32_t(count);
            return asset;
        }

    private:
        static bool hasExtension(const std::string &s, std::initializer_list<std::string> exts) {
            auto ext = std::filesystem::path(s).extension().string();
            return std::any_of(exts.begin(), exts.end(), [&](auto &e) { return e == ext; });
        }

        static std::shared_ptr<tinyply::PlyData> tryProperties(
            tinyply::PlyFile &ply,
            const std::string &elem,
            const std::vector<std::string> &names) {
            try { return ply.request_properties_from_element(elem, names); } catch (...) { return nullptr; }
        }

        static void validateCount(
            const std::shared_ptr<tinyply::PlyData> &d,
            size_t count,
            const std::string &name,
            const std::filesystem::path &p) {
            if (d && d->count != count)
                throw std::runtime_error(name + " count mismatch in " + p.string());
        }
    };

    struct RandomCloudRequest {
        glm::vec3 minCorner; // world-space origin (lower-left-front)
        glm::vec3 size; // {width, height, depth}  in metres
        float density; // gaussians / m³
        int maxPoints{0}; // 0 = unlimited  (optional cap)
        int seed{0}; // 0 = std::random_device (optional)
    };

    static std::shared_ptr<Gaussian2DAsset>
    createRandomGaussianCloud(const RandomCloudRequest &req) {
        /* ------------------------------------------------------------
         *  RNG
         * ---------------------------------------------------------- */
        std::mt19937 rng(req.seed ? req.seed : std::random_device{}());

        /* ------------------------------------------------------------
         *  Decide how many points
         * ---------------------------------------------------------- */
        const float volume = req.size.x * req.size.y * req.size.z; // m³
        uint32_t count = static_cast<uint32_t>(volume * req.density);
        if (req.maxPoints && count > req.maxPoints) // optional cap
            count = req.maxPoints;

        /* ------------------------------------------------------------
         *  Distributions for attributes
         * ---------------------------------------------------------- */
        std::uniform_real_distribution<float>
                ux(0.0f, req.size.x),
                uy(0.0f, req.size.y),
                uz(0.0f, req.size.z),
                col(0.2, 0.90f),
                opa(0.10f, 1.30f); // pre-sigmoid

        std::normal_distribution<float> logScale(std::log(0.30f), 0.15f);

        /* ------------------------------------------------------------
         *  Build the asset
         * ---------------------------------------------------------- */
        auto asset = std::make_shared<Gaussian2DAsset>();
        asset->reserve(count);

        std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
        auto randQuat = [&]() {
            // Shoemake’s algorithm for uniform quaternion
            float u1 = uDist(rng);
            float u2 = uDist(rng);
            float u3 = uDist(rng);

            float r1 = std::sqrt(1.0f - u1);
            float r2 = std::sqrt(u1);

            float theta1 = 2.0f * glm::pi<float>() * u2;
            float theta2 = 2.0f * glm::pi<float>() * u3;

            float w = std::cos(theta2) * r2;
            float x = std::sin(theta1) * r1;
            float y = std::cos(theta1) * r1;
            float z = std::sin(theta2) * r2;

            return glm::quat(w, x, y, z);
        };

        for (uint32_t i = 0; i < count; ++i) {
            /* ---- position ---- */
            glm::vec3 p{ux(rng), uy(rng), uz(rng)};
            asset->positions.emplace_back(req.minCorner + p);

            /* ---- rotation : yaw only ---- */

            asset->rotations.emplace_back(randQuat());


            /* ---- scale (log-space stored) ---- */
            float sx = std::clamp(logScale(rng), -3.0f, 0.0f); // ~exp 0.05–1.0
            float sy = std::clamp(logScale(rng), -3.0f, 0.0f);
            asset->scale_x.push_back(std::exp(sx));
            asset->scale_y.push_back(std::exp(sy));

            /* ---- colour ---- */
            float color = col(rng);
            asset->colors.emplace_back(color, color, color);

            /* ---- opacity (logistic) ---- */
            float alpha = 1.f / (1.f + std::exp(-opa(rng)));
            asset->opacity.push_back(alpha);
        }

        asset->numPoints = count;
        return asset;
    }
}

#endif // TWODGPOINTCLOUDLOADER_H
