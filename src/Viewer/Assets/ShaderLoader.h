//
// Created by mgjer on 27/04/2025.
//

#ifndef SHADERLOADER_H
#define SHADERLOADER_H

#include <filesystem>

#include "Viewer/Assets/IAssetLoader.h"
#include <vulkan/vulkan.h>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace VkRender {

    struct SPIRVAsset : BaseAsset {
        std::vector<uint32_t> code;

        // Construct directly from a word‐vector
        explicit SPIRVAsset(std::vector<uint32_t>&& spirv)
          : code(std::move(spirv)) {}
    };

    class ShaderLoader : public IAssetLoader {
    public:
        std::type_index assetType() const override {return typeid(SPIRVAsset);}

        explicit ShaderLoader() = default;
        bool canLoad(const std::string& key) const override;
        std::shared_ptr<BaseAsset> load(const std::filesystem::path&) override;

    private:
        std::unordered_map<std::string, std::shared_ptr<SPIRVAsset>> m_moduleCache;
        std::mutex m_mutex;
    };
}

#endif //SHADERLOADER_H
