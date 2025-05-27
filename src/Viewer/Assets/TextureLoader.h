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
            int w, h, c;
            std::filesystem::path path = Utils::getTexturePath() / key;
            stbi_uc *data = stbi_load(path.string().c_str(), &w, &h, &c, STBI_rgb_alpha);
            if (!data)
                throw std::runtime_error("Failed to load Texture " + path.string());
            c = 4; // TODO Force 4 channels, possibly make more flexible in the future if required for more difficult textures
            std::vector<uint8_t> pixels(data, data + (w * h * c));
            stbi_image_free(data);
            return std::make_shared<TextureAsset>(w, h, c, std::move(pixels));
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
#endif //TEXTURELOADER_H
