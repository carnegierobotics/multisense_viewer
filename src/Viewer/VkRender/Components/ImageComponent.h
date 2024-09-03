//
// Created by mgjer on 18/08/2024.
//

#ifndef MULTISENSE_VIEWER_IMAGECOMPONENT_H
#define MULTISENSE_VIEWER_IMAGECOMPONENT_H

#include <stb_image.h>

#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/Tools/Logger.h"

namespace VkRender {
    struct   ImageComponent {
    public:


        ImageComponent() = delete;

        ImageComponent(const ImageComponent &) = delete;

        ImageComponent &operator=(const ImageComponent &other) {
            return *this;
        }

        explicit ImageComponent(const std::filesystem::path &imagePath) {
            m_vertices = {
                    // Bottom-left corner
                    {glm::vec2{-1.0f, -1.0f}, glm::vec2{0.0f, 0.0f}},
                    // Bottom-right corner
                    {glm::vec2{1.0f, -1.0f},  glm::vec2{1.0f, 0.0f}},
                    // Top-right corner
                    {glm::vec2{1.0f, 1.0f},   glm::vec2{1.0f, 1.0f}},
                    // Top-left corner
                    {glm::vec2{-1.0f, 1.0f},  glm::vec2{0.0f, 1.0f}}
            };
            // Define the indices for two triangles that make up the quad
            m_indices = {
                    0, 1, 2, // First triangle (bottom-left to top-right)
                    2, 3, 0  // Second triangle (top-right to bottom-left)
            };
            m_texPath = imagePath;
            loadTexture();
        }
        ImageComponent(uint32_t width, uint32_t height) {
            m_vertices = {
                    // Bottom-left corner
                    {glm::vec2{-1.0f, -1.0f}, glm::vec2{0.0f, 0.0f}},
                    // Bottom-right corner
                    {glm::vec2{1.0f, -1.0f},  glm::vec2{1.0f, 0.0f}},
                    // Top-right corner
                    {glm::vec2{1.0f, 1.0f},   glm::vec2{1.0f, 1.0f}},
                    // Top-left corner
                    {glm::vec2{-1.0f, 1.0f},  glm::vec2{0.0f, 1.0f}}
            };
            // Define the indices for two triangles that make up the quad
            m_indices = {
                    0, 1, 2, // First triangle (bottom-left to top-right)
                    2, 3, 0  // Second triangle (top-right to bottom-left)
            };


            m_texSize = STBI_rgb_alpha * height * width;
            m_texHeight = height;
            m_texWidth = width;

        }

        ~ImageComponent() {
            if (m_pixels) {
                stbi_image_free(m_pixels);
            }
        }

        [[nodiscard]] stbi_uc *getTexture() const { return m_pixels; }
        [[nodiscard]] const std::filesystem::path& getTextureFilePath() const { return m_texPath; }
        [[nodiscard]] VkDeviceSize getTextureSize() const { return m_texSize; }

    private:
        void loadTexture() {
            int texWidth = 0, texHeight = 0, texChannels = 0;
            m_pixels = stbi_load(m_texPath.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            if (!m_pixels) {
                Log::Logger::getInstance()->error("Failed to load texture: {}", m_texPath.string());

            }
            m_texSize = STBI_rgb_alpha * texHeight * texWidth;
            m_texHeight = texHeight;
            m_texWidth = texWidth;
        }

    public:
        // The quad which the image is displayed on
        std::vector<VkRender::ImageVertex> m_vertices;
        std::vector<uint32_t> m_indices;
        // The texture
        std::filesystem::path m_texPath;
        stbi_uc *m_pixels = nullptr;
        uint32_t m_texWidth = 0;
        uint32_t m_texHeight = 0;
        VkDeviceSize m_texSize = 0;
    };

}

#endif //MULTISENSE_VIEWER_IMAGECOMPONENT_H
