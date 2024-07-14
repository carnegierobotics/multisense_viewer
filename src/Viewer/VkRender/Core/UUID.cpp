//
// Created by magnus on 4/11/24.
//

#include "Viewer/VkRender/Core/UUID.h"

#include <random>
#include <unordered_map>

namespace VkRender {

    static std::random_device s_RandomDevice;
    static std::mt19937_64 s_Engine(s_RandomDevice());
    static std::uniform_int_distribution<uint64_t> s_UniformDistribution;

    UUID::UUID()
            : m_UUID(s_UniformDistribution(s_Engine))
    {
    }

    UUID::UUID(uint64_t uuid)
            : m_UUID(uuid)
    {
    }

}