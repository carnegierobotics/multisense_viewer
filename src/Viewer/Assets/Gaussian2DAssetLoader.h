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

namespace VkRender {
    //--------------------------------------------------------------------------
    // Asset holding 2‑D Gaussian splat data
    //--------------------------------------------------------------------------
    struct Gaussian2DAsset : BaseAsset {
        std::vector<glm::vec3> positions;    // mean µ
        std::vector<glm::quat> rotations;    // orientation of the tangent plane (w, x, y, z)
        std::vector<float>     scale_x;      // σ_x
        std::vector<float>     scale_y;      // σ_y
        std::vector<glm::vec3> colors;       // zero‑order SH (linear RGB)
        std::vector<float>     opacity;      // α ∈ [0,1]
        std::vector<std::vector<float>> shCoeffs; // higher-order SH (per-point)

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
            return hasExtension(key, { ".ply" });
        }

        std::type_index assetType() const override {
            return typeid(Gaussian2DAsset);
        }

        std::shared_ptr<BaseAsset> load(const std::filesystem::path& key) override {
            using tinyply::PlyData;
            std::ifstream stream(key, std::ios::binary);
            if (!stream.is_open())
                throw std::runtime_error("Failed to open Gaussian2DAsset: " + key.string());

            tinyply::PlyFile ply;
            ply.parse_header(stream);

            // Mandatory
            auto posData = ply.request_properties_from_element("vertex", {"x","y","z"});
            // Optional unary properties
            auto rotData = tryProperties(ply,"vertex", {"rot_0","rot_1","rot_2","rot_3"});
            auto sxData  = tryProperties(ply,"vertex", {"scale_0"});
            auto syData  = tryProperties(ply,"vertex", {"scale_1"});
            auto colData = tryProperties(ply,"vertex", {"f_dc_0","f_dc_1","f_dc_2"});
            auto opaData = tryProperties(ply,"vertex", {"opacity"});


            ply.read(stream);

            const size_t count = posData->count;
            validateCount(rotData, count, "rot_*", key);
            validateCount(sxData,  count, "scale_0", key);
            validateCount(syData,  count, "scale_1", key);
            validateCount(colData, count, "f_dc_*", key);
            validateCount(opaData, count, "opacity", key);

            // Copy buffers
            std::vector<float> posBuf(count*3);
            std::memcpy(posBuf.data(), posData->buffer.get(), posBuf.size()*sizeof(float));

            auto copyOrDefault = [&](std::shared_ptr<PlyData> &src, float def) {
                std::vector<float> dst(count, def);
                if (src) std::memcpy(dst.data(), src->buffer.get(), count*sizeof(float));
                return dst;
            };
            auto copyOrDefaultVec4 = [&](std::shared_ptr<PlyData> &src, glm::quat def) {
                std::vector<float> dst(count*4);
                if (src) std::memcpy(dst.data(), src->buffer.get(), dst.size()*sizeof(float));
                else for (size_t i=0;i<count;++i) {
                    dst[i*4+0]=def.w; dst[i*4+1]=def.x;
                    dst[i*4+2]=def.y; dst[i*4+3]=def.z;
                }
                return dst;
            };

            std::vector<float> rotBuf = copyOrDefaultVec4(rotData, glm::quat{1,0,0,0});
            std::vector<float> sxBuf  = copyOrDefault(sxData, 1.f);
            std::vector<float> syBuf  = copyOrDefault(syData, 1.f);
            std::vector<float> colBuf = [&](){
                std::vector<float> d(count*3,1.f);
                if (colData) std::memcpy(d.data(), colData->buffer.get(), d.size()*sizeof(float));
                return d;
            }();
            std::vector<float> opaBuf = copyOrDefault(opaData, 1.f);


            auto asset = std::make_shared<Gaussian2DAsset>();
            asset->reserve(count);

            for (size_t i=0;i<count;++i) {
                asset->positions.emplace_back(
                    posBuf[i*3+0], posBuf[i*3+1], posBuf[i*3+2]
                );
                asset->rotations.emplace_back(
                    glm::normalize(glm::quat(rotBuf[i*4+0], rotBuf[i*4+1], rotBuf[i*4+2], rotBuf[i*4+3]))
                );
                asset->scale_x.push_back(std::exp(sxBuf[i]));
                asset->scale_y.push_back(std::exp(syBuf[i]));
                asset->colors.emplace_back(
                    colBuf[i*3+0], colBuf[i*3+1], colBuf[i*3+2]
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
            return std::any_of(exts.begin(), exts.end(), [&](auto &e){ return e==ext; });
        }

        static std::shared_ptr<tinyply::PlyData> tryProperties(
            tinyply::PlyFile &ply,
            const std::string &elem,
            const std::vector<std::string> &names)
        {
            try { return ply.request_properties_from_element(elem, names); }
            catch(...) { return nullptr; }
        }

        static void validateCount(
            const std::shared_ptr<tinyply::PlyData> &d,
            size_t count,
            const std::string &name,
            const std::filesystem::path &p)
        {
            if (d && d->count!=count)
                throw std::runtime_error(name + " count mismatch in " + p.string());
        }
    };
}

#endif // TWODGPOINTCLOUDLOADER_H
