//
// Created by mgjer on 27/04/2025.
//

#ifndef SHADERLOADER_H
#define SHADERLOADER_H

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
        explicit ShaderLoader(VkDevice device) {

        }
        bool canLoad(const std::string& key) const override;
        std::shared_ptr<BaseAsset> load(const std::string& key) override;

    private:
        std::unordered_map<std::string, std::shared_ptr<SPIRVAsset>> m_moduleCache;
        std::mutex m_mutex;
    };
}

#endif //SHADERLOADER_H
