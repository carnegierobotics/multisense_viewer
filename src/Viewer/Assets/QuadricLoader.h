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
        std::vector<float> a, b, c;
        std::vector<float> t_x, t_y;
        std::vector<float> kernelScale, threshold, beta;

        uint32_t numPoints = 0;

        // Reserve for n quadrics
        void reserve(size_t n) {
            positions.reserve(n);
            rotations.reserve(n);
            a.reserve(n);
            b.reserve(n);
            c.reserve(n);
            t_x.reserve(n);
            t_y.reserve(n);
            kernelScale.reserve(n);
            threshold.reserve(n);
            beta.reserve(n);
        }
    };

    // Loader for .ply quadric‐point clouds
    class QuadricLoader : public IAssetLoader {
    public:
        bool canLoad(const std::string &key) const override {
            return hasExtension(key, {".ply"});
        }

        std::shared_ptr<BaseAsset> load(const std::string &key) override {
            std::filesystem::path path{key};
            std::ifstream stream(path, std::ios::binary);
            if (!stream.is_open())
                throw std::runtime_error("Failed to open QuadricAsset: " + path.string());

            tinyply::PlyFile ply;
            ply.parse_header(stream);

            //----------------------------------------------------------------------
            // 1.  Mandatory properties
            //----------------------------------------------------------------------
            auto posData = ply.request_properties_from_element("vertex", {"x", "y", "z"});

            //----------------------------------------------------------------------
            // 2.  Optional properties (request with required = false)
            //
            //     If the property is not present tinyply returns a nullptr, which
            //     we detect later and replace with a vector of defaults.
            //----------------------------------------------------------------------

            auto try_request_prop = [&](const std::initializer_list<std::string> &names)
                -> std::shared_ptr<tinyply::PlyData> {
                try {
                    return ply.request_properties_from_element("vertex", names);
                } catch (const std::exception &) // property or element not found
                {
                    return nullptr;
                }
            };

            auto rotData = try_request_prop({"rot_0", "rot_1", "rot_2", "rot_3"});
            auto aData = try_request_prop({"a"});
            auto bData = try_request_prop({"b"});
            auto cData = try_request_prop({"c"});
            auto txData = try_request_prop({"t_x"});
            auto tyData = try_request_prop({"t_y"});
            auto ksData = try_request_prop({"kernel_scale"});
            auto thrData = try_request_prop({"threshold"});
            auto betaData = try_request_prop({"beta"});

            ply.read(stream);

            //----------------------------------------------------------------------
            // 3.  Vertex count and quick sanity check on rotations
            //----------------------------------------------------------------------
            const std::size_t count = posData->count;


            //----------------------------------------------------------------------
            // 4.  Copy raw buffers (mandatory) ------------------------------------
            //----------------------------------------------------------------------
            std::vector<float> posBuf(count * 3);
            std::memcpy(posBuf.data(), posData->buffer.get(), count * 3 * sizeof(float));

            //----------------------------------------------------------------------
            // 5.  Helper: allocate vector<T>(count, defaultValue) -----------------
            //----------------------------------------------------------------------
            auto make_default = [count](float def) {
                return std::vector<float>(count, def);
            };

            //----------------------------------------------------------------------
            // 6.  Optional buffers (present ? copy : fill with defaults) ----------
            //----------------------------------------------------------------------
            std::default_random_engine rng(std::random_device{}());

            auto copy_or_default = [&](std::shared_ptr<tinyply::PlyData> &src,
                                       float def,
                                       float noise_amplitude = 0.0f) -> std::vector<float> {
                std::vector<float> dst(count);
                if (src) {
                    if (src->count != count)
                        throw std::runtime_error("Inconsistent vertex count for optional "
                                                 "property in: " + path.string());
                    std::memcpy(dst.data(), src->buffer.get(), count * sizeof(float));
                } else {
                    std::normal_distribution dist(-noise_amplitude, noise_amplitude);
                    for (auto &v : dst)
                        v = def + dist(rng);
                }
                return dst;
            };

            // 2.b NEW helper for vec4-per-vertex
            auto copy_or_default_vec4 = [&](std::shared_ptr<tinyply::PlyData> &src,
                                            const std::array<float,4> &def) -> std::vector<float>
            {
                std::vector<float> dst(count * 4);
                if (src)
                {
                    if (src->count != count)
                        throw std::runtime_error("Inconsistent vertex count in: " + path.string());
                    std::memcpy(dst.data(), src->buffer.get(), count * 4 * sizeof(float));
                }
                else
                {
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        std::memcpy(dst.data() + i*4, def.data(), 4*sizeof(float));
                    }
                }
                return dst;
            };

            // 3. build buffers ------------------------------------------------
            const std::vector<float> rotBuf = copy_or_default_vec4(rotData, {1.0f,0.0f,0.0f,0.0f});

            const std::vector<float> aBuf   = copy_or_default(aData, 1.0f, 0.05f);
            const std::vector<float> bBuf   = copy_or_default(bData, 1.0f, 0.05f);
            const std::vector<float> cBuf   = copy_or_default(cData, 1.0f, 0.05f);
            const std::vector<float> txBuf  = copy_or_default(txData, -1.0f, 0.1f);
            const std::vector<float> tyBuf  = copy_or_default(tyData, 1.0f, 0.1f);
            const std::vector<float> ksBuf  = copy_or_default(ksData, 0.25f, 0.02f);
            const std::vector<float> thrBuf = copy_or_default(thrData, 0.1f, 0.01f);
            const std::vector<float> betBuf = copy_or_default(betaData, 0.0f, 0.01f);

            //----------------------------------------------------------------------
            // 7.  Build the asset --------------------------------------------------
            //----------------------------------------------------------------------
            auto asset = std::make_shared<QuadricCloudAsset>();
            asset->reserve(count);

            for (std::size_t i = 0; i < count; ++i) {
                // position
                asset->positions.emplace_back(
                    posBuf[i * 3 + 0],
                    posBuf[i * 3 + 1],
                    posBuf[i * 3 + 2]);

                // rotation (w,x,y,z)
                asset->rotations.emplace_back(
                    rotBuf[i * 4 + 0],
                    rotBuf[i * 4 + 1],
                    rotBuf[i * 4 + 2],
                    rotBuf[i * 4 + 3]);

                // quadric & kernel parameters
                asset->a.push_back(aBuf[i]);
                asset->b.push_back(bBuf[i]);
                asset->c.push_back(cBuf[i]);
                asset->t_x.push_back(txBuf[i]);
                asset->t_y.push_back(tyBuf[i]);
                asset->kernelScale.push_back(ksBuf[i]);
                asset->threshold.push_back(thrBuf[i]);
                asset->beta.push_back(betBuf[i]);
            }

            asset->numPoints = count;
            return asset;
        }

    private:
        // Helper to check file extension
        bool hasExtension(const std::string &s, std::initializer_list<std::string> exts) const {
            auto ext = std::filesystem::path(s).extension().string();
            for (auto &e: exts) if (ext == e) return true;
            return false;
        }
    };
}


#endif //QUADRICPOINTCLOUDLOADER_H
