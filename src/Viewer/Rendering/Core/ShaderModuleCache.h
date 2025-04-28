// ShaderModuleCache.h
#pragma once
#include "VulkanShaderModule.h"
#include "Viewer/Assets/ShaderLoader.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace VkRender {

    class ShaderModuleCache {
    public:
        explicit ShaderModuleCache(VkDevice device)
          : _device(device)
        {}

        std::shared_ptr<VulkanShaderModule>
        get(const VulkanShaderModuleCreateInfo& ci) {
            Key key{ _device, ci.spirvAsset.get(), ci.shaderStage, ci.debugInfo };

            std::lock_guard<std::mutex> lk(_mutex);
            auto it = _cache.find(key);
            if (it != _cache.end()) {
                    return it->second;
            }

            auto module = std::make_shared<VulkanShaderModule>(ci);
            _cache[key] = module;
            return module;
        }

    private:
        struct Key {
            VkDevice                device;
            SPIRVAsset*             spirvPtr;
            VkShaderStageFlagBits   stage;
            std::string             debugInfo;
        };
        struct Hash {
            size_t operator()(Key const& k) const noexcept {
                auto h1 = std::hash<void*>()((void*)k.device);
                auto h2 = std::hash<void*>()((void*)k.spirvPtr);
                auto h3 = std::hash<uint32_t>()((uint32_t)k.stage);
                auto h4 = std::hash<std::string>()(k.debugInfo);
                return h1 ^ (h2<<1) ^ (h3<<2) ^ (h4<<3);
            }
        };
        struct Eq {
            bool operator()(Key const& a, Key const& b) const noexcept {
                return a.device    == b.device
                    && a.spirvPtr  == b.spirvPtr
                    && a.stage     == b.stage
                    && a.debugInfo == b.debugInfo;
            }
        };

        VkDevice _device;
        std::mutex    _mutex;
        std::unordered_map<Key,
                           std::shared_ptr<VulkanShaderModule>,
                           Hash, Eq> _cache;
    };

} // namespace VkRender
