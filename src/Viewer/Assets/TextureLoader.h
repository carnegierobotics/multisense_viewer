//
// Created by magnus on 4/28/25.
//

#ifndef TEXTURELOADER_H
#define TEXTURELOADER_H
#include <stb_image.h>
#include <vector>
#include <filesystem>

#include "IAssetLoader.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {
    // CPUTextureAsset.h
    struct TextureAsset : BaseAsset {
        int width = 0;
        int height = 0;
        int depth = 1;
        int channels = 0;
        std::vector<uint8_t> pixels;

        TextureAsset(int w, int h, int c, std::vector<uint8_t> &&p)
            : width(w), height(h), channels(c), pixels(std::move(p)) {
        }
    };

    // CPUTextureLoader.h
    class TextureLoader : public IAssetLoader {
    public:
        bool canLoad(const std::string &key) const override {
            return hasExtension(key, {".png", ".jpg", ".hdr"});
        }
        std::type_index assetType() const override {return typeid(TextureAsset);}


        std::shared_ptr<BaseAsset> load(const std::filesystem::path& key) override {
            const auto keyStr = key.string();

            // 1) Check cache under lock
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_textureCache.find(keyStr);
                if (it != m_textureCache.end()) {
                    return it->second;
                }
            }

            // 2) Build full path
            std::filesystem::path path = Utils::getTexturePath() / key;
            if (!std::filesystem::exists(path)) {
                Log::Logger::getInstance()->error("TextureLoader: file not found: {}", path.string());
                return nullptr;
            }

            // 3) Load pixels from disk
            int w, h, c;
            stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &c, STBI_rgb_alpha);
            if (!data) {
                Log::Logger::getInstance()->error("TextureLoader: failed to load image {}", path.string());
                return nullptr;
            }
            c = 4; // force RGBA
            size_t size = static_cast<size_t>(w) * h * c;
            std::vector<uint8_t> pixels(data, data + size);
            stbi_image_free(data);

            // 4) Wrap in asset
            auto asset = std::make_shared<TextureAsset>(w, h, c, std::move(pixels));

            // 5) Cache and return
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_textureCache.emplace(keyStr, asset);
            }
            return asset;
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<BaseAsset>> m_textureCache;
        std::mutex m_mutex;

        // Helper to check file extension
        bool hasExtension(const std::string &s, std::initializer_list<std::string> exts) const {
            auto ext = std::filesystem::path(s).extension().string();
            for (auto &e : exts) if (ext == e) return true;
            return false;
        }
    };
}
#endif //TEXTURELOADER_H
