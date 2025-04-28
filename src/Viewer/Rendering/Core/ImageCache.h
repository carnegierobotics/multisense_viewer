//
// Created by magnus on 4/28/25.
//

#ifndef IMAGECACHE_H
#define IMAGECACHE_H

#include <unordered_map>
#include <mutex>
#include <functional>    // for std::hash
#include <cstdint>       // for uintptr_t
#include <string>


namespace VkRender {

class VulkanImageCache {
public:
    explicit VulkanImageCache(VkDevice device)
      : m_device(device)
    {}

    std::shared_ptr<VulkanImage>
    get(const VulkanImageCreateInfo& ci) {
        // build a key from all the things that make this image unique:
        Key key{
            m_device,
            &ci.allocator,
            ci.imageCreateInfo.format,
            ci.imageCreateInfo.extent,
            ci.imageCreateInfo.usage,
            ci.imageViewCreateInfo.subresourceRange.aspectMask,
            ci.setLayout,
            ci.srcLayout,
            ci.dstLayout
        };

        std::lock_guard<std::mutex> lk(m_mutex);
        auto it = m_cache.find(key);
        if (it != m_cache.end())
            return it->second;

        // miss → construct, cache, and return
        auto img = std::make_shared<VulkanImage>(const_cast<VulkanImageCreateInfo&>(ci));
        m_cache.emplace(std::move(key), img);
        return img;
    }

private:
    struct Key {
        VkDevice              device;
        VmaAllocator*         allocator;
        VkFormat              format;
        VkExtent3D            extent;
        VkImageUsageFlags     usage;
        VkImageAspectFlags    aspectMask;
        bool                  setLayout;
        VkImageLayout         srcLayout;
        VkImageLayout         dstLayout;

        bool operator==(Key const& o) const noexcept {
            return device      == o.device
                && allocator   == o.allocator
                && format      == o.format
                && extent.width== o.extent.width
                && extent.height==o.extent.height
                && extent.depth== o.extent.depth
                && usage       == o.usage
                && aspectMask  == o.aspectMask
                && setLayout   == o.setLayout
                && srcLayout   == o.srcLayout
                && dstLayout   == o.dstLayout;
        }
    };

    struct Hash {
        size_t operator()(Key const& k) const noexcept {
            // Start with a hash of the device handle
            size_t h = std::hash<VkDevice>()(k.device);
            // Combine in the allocator pointer
            h ^= std::hash<std::uintptr_t>()((std::uintptr_t)k.allocator)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            // Combine in the format
            h ^= std::hash<uint32_t>()((uint32_t)k.format)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            // Combine in each dimension
            h ^= std::hash<uint32_t>()(k.extent.width)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= std::hash<uint32_t>()(k.extent.height)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= std::hash<uint32_t>()(k.extent.depth)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            // Usage & aspect
            h ^= std::hash<uint32_t>()((uint32_t)k.usage)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= std::hash<uint32_t>()((uint32_t)k.aspectMask)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            // Layout flags
            h ^= std::hash<bool>()(k.setLayout)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= std::hash<uint32_t>()((uint32_t)k.srcLayout)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            h ^= std::hash<uint32_t>()((uint32_t)k.dstLayout)
               + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);

            return h;
        }
    };

    VkDevice   m_device;
    std::mutex m_mutex;
    std::unordered_map<Key, std::shared_ptr<VulkanImage>, Hash> m_cache;
};

}
#endif //IMAGECACHE_H
