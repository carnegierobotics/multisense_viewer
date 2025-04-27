//
// Created by mgjer on 27/04/2025.
//


#include "Viewer/Assets/ShaderLoader.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {

    bool ShaderLoader::canLoad(const std::string& key) const {
        return key.ends_with(".spv") ||  key.ends_with(".comp") ||  key.ends_with(".frag") ||  key.ends_with(".vert");
    }

    std::shared_ptr<BaseAsset> ShaderLoader::load(const std::string& key) {
        std::lock_guard<std::mutex> lock(m_mutex);

        // 1) Cache lookup
        auto it = m_moduleCache.find(key);
        if (it != m_moduleCache.end()) {
            return it->second;
        }

        // 2) Build filesystem path
        auto path = Utils::getShadersPath() / key;
        if (path.extension() != ".spv") {
            path += ".spv";
        }

        // 3) Read SPIR-V bytes
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            Log::Logger::getInstance()->error("ShaderLoader: failed to open {}", path.string());
            return nullptr;
        }
        size_t      size = static_cast<size_t>(file.tellg());
        std::vector<uint32_t> spirv(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(spirv.data()), size);
        file.close();

        // 4) Wrap and cache
        auto asset = std::make_shared<SPIRVAsset>(std::move(spirv));
        m_moduleCache.emplace(key, asset);
        return asset;
    }

} // namespace VkRender