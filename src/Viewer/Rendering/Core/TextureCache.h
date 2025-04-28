//
// Created by magnus on 4/28/25.
//

#ifndef TEXTURECACHE_H
#define TEXTURECACHE_H

#include <Viewer/Assets/TextureLoader.h>
#include <Viewer/Rendering/Core/VulkanTexture.h>

namespace VkRender {

    class TextureCache {
    public:
        explicit TextureCache(VkDevice device)
          : m_device(device)
        {}

        std::shared_ptr<VulkanTexture2D>
        get(const VulkanTexture2DCreateInfo &ci) {
            // Key is (device + raw VulkanImage pointer).
            Key key{ m_device, ci.image.get() };

            std::lock_guard<std::mutex> lk(m_mutex);
            auto it = m_cache.find(key);
            if (it != m_cache.end())
                return it->second;

            // Miss: build one, stash it, and return it.
            auto tex = std::make_shared<VulkanTexture2D>(const_cast<VulkanTexture2DCreateInfo&>(ci));
            m_cache.emplace(key, tex);
            return tex;
        }

    private:
        struct Key {
            VkDevice       device;
            VulkanImage   *imagePtr;

            bool operator==(Key const &o) const noexcept {
                return device == o.device
                    && imagePtr == o.imagePtr;
            }
        };

        struct Hash {
            size_t operator()(Key const &k) const noexcept {
                // mix the device-handle + pointer bits
                auto h1 = std::hash<VkDevice>()(k.device);
                auto h2 = std::hash<std::uintptr_t>()((std::uintptr_t)k.imagePtr);
                return h1 ^ (h2 << 1);
            }
        };

        VkDevice   m_device;
        std::mutex m_mutex;
        std::unordered_map<Key,
                           std::shared_ptr<VulkanTexture2D>,
                           Hash> m_cache;
    };


}

#endif //TEXTURECACHE_H
