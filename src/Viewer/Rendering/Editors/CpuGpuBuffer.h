//
// Created by magnus on 4/28/25.
//

#ifndef CPUGPUBUFFER_H
#define CPUGPUBUFFER_H

namespace VkRender {
    // ─────────────────────────────────────────────────────────────────────────────
    // 3.  Tiny wrapper around a per‑frame CPU→GPU buffer
    // ─────────────────────────────────────────────────────────────────────────────
    template<VkBufferUsageFlags USAGE>
    class CpuGpuBuffer {
    public:
        template<typename T>
        void init(VulkanDevice &dev,
                  uint32_t frames,
                  uint32_t maxItems,
                  const char *dbgName) {
            m_frames = frames;
            m_maxItems = maxItems;
            m_buffers.resize(frames);
            VkDeviceSize bytes = sizeof(T) * maxItems;
            for (uint32_t f = 0; f < frames; ++f) {
                dev.createBuffer(USAGE,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 m_buffers[f], bytes, nullptr, dbgName,
                                 nullptr);
            }
        }

        // upload [count] items from src into frame f
        void upload(VulkanDevice &dev, uint32_t f,
                    const void *src, uint32_t count, VkDeviceSize itemSize) {
            assert(count <= m_maxItems && "buffer overflow — raise kMax* constant");
            if (count == 0 || itemSize == 0)
                return;

            void *dst;
            vkMapMemory(dev.m_LogicalDevice, m_buffers[f]->m_memory,
                        0, itemSize * count, 0, &dst);
            std::memcpy(dst, src, itemSize * count);
            vkUnmapMemory(dev.m_LogicalDevice, m_buffers[f]->m_memory);
        }

        VkDescriptorBufferInfo info(uint32_t f, VkDeviceSize range) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = m_buffers[f]->m_buffer;
            bi.offset = 0;
            bi.range = range;
            return bi;
        }

        Buffer *raw(uint32_t f) { return m_buffers[f].get(); }

    private:
        uint32_t m_frames = 0;
        uint32_t m_maxItems = 0;
        std::vector<std::unique_ptr<Buffer> > m_buffers;
    };

    using CpuUniformBuffer = CpuGpuBuffer<VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT>;
    using CpuStorageBuffer = CpuGpuBuffer<VK_BUFFER_USAGE_STORAGE_BUFFER_BIT>;
}


#endif //CPUGPUBUFFER_H
