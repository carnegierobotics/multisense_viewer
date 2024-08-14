//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_OBJMODELCOMPONENT
#define MULTISENSE_VIEWER_OBJMODELCOMPONENT

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <cfloat>
#include <stb_image.h>

#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"

namespace VkRender {
    struct OBJModelComponent {
    public:
        OBJModelComponent() = default;

        OBJModelComponent(const OBJModelComponent &) = delete;

        OBJModelComponent &operator=(const OBJModelComponent &other) {
            return *this;
        }

        explicit OBJModelComponent(const std::filesystem::path &modelPath) {
            loadModel(modelPath);
            loadTexture(modelPath);
        }

        Texture2D objTexture; // TODO Possibly make more empty textures to match our triple buffering?

    private:
        void loadModel(const std::filesystem::path& modelPath);

        void loadTexture(const std::filesystem::path& texturePath);

    public:
        std::vector<VkRender::Vertex> m_vertices;
        std::vector<uint32_t> m_indices;
        stbi_uc *m_pixels{};
        VkDeviceSize m_texSize = 0;
        uint32_t m_texWidth = 0;
        uint32_t m_texHeight = 0;
    };
};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT
