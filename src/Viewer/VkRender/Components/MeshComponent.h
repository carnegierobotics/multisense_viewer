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
    struct MeshComponent {
    public:


        MeshComponent() = default;

        MeshComponent(const MeshComponent &) = delete;

        MeshComponent &operator=(const MeshComponent &other) {
            return *this;
        }

        explicit MeshComponent(const std::filesystem::path &modelPath) {
            loadModel(modelPath);
            loadTexture(modelPath);
            m_modelPath = modelPath;
            m_initialized = true;
        }

        explicit MeshComponent(uint32_t type) {
            loadCameraModelMesh();
            m_modelPath = "Generated";
            m_initialized = true;
        }

        Texture2D objTexture; // TODO Possibly make more empty textures to match our triple buffering?
        UBOCamera &getCameraModelMesh() { return m_cameraModelVertices; }
        bool usesUBOMesh() {return m_isCameraModelMesh;}
        bool hasMesh() {return m_initialized;}
        void loadOBJ(std::filesystem::path modelPath);

        std::filesystem::path getModelPath() {return m_modelPath;}
    private:
        void loadModel(const std::filesystem::path &modelPath);

        void loadTexture(const std::filesystem::path &texturePath);


        void loadCameraModelMesh();


        UBOCamera m_cameraModelVertices{};

    public:
        std::vector<VkRender::Vertex> m_vertices;
        std::vector<uint32_t> m_indices;
        stbi_uc *m_pixels{};
        VkDeviceSize m_texSize = 0;
        uint32_t m_texWidth = 0;
        uint32_t m_texHeight = 0;
        std::filesystem::path m_modelPath;

        bool m_isCameraModelMesh = false;
        bool m_initialized = false;

    };
};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT
