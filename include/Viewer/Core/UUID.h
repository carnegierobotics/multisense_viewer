// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_UUID_H
#define MULTISENSE_VIEWER_UUID_H

#include <cstdint> // Include for uint64_t
#include <string> // Include for uint64_t

namespace VkRender {

    class UUID
    {
    public:
        UUID();
        explicit UUID(uint64_t uuid);
        UUID(const UUID&) = default;

        operator uint64_t() const { return m_UUID; }
        operator std::string() { return std::to_string(m_UUID); }
    private:
        uint64_t m_UUID;
    };

}

namespace std {
    template <typename T> struct hash;

    template<>
    struct hash<VkRender::UUID>
    {
        std::size_t operator()(const VkRender::UUID& uuid) const
        {
            return static_cast<uint64_t>(uuid);
        }
    };

}

#endif //MULTISENSE_VIEWER_UUID_H
