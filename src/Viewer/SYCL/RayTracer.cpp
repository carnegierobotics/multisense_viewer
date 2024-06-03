//
// Created by mgjer on 03/06/2024.
//
#include <random>

#include "RayTracer.h"

namespace VkRender {

    void RayTracer::setup(const InitializeInfo &initInfo) {
        m_initInfo = initInfo;
        m_image = static_cast<uint8_t *>(malloc(initInfo.imageSize));
    }

    void RayTracer::render(const RenderInfo &renderInfo) {
        // Seed with a real random value, if available
        std::random_device rd;

        // Initialize a random number generator
        std::mt19937 gen(rd());

        // Define a distribution from 0 to 255 inclusive
        std::uniform_int_distribution<> distrib(0, 25);

        // Generate a random number
        int random_value = distrib(gen);
        memset(m_image, random_value, m_initInfo.imageSize);

    }

    uint8_t *RayTracer::getImage() {
        return m_image;
    }

    uint32_t RayTracer::getImageSize() {
        return m_initInfo.imageSize;
    }
}