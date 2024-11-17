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
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Core/UUID.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {
    enum MeshDataType {
        OBJ_FILE,
        POINT_CLOUD,
        PLY_FILE,
        CAMERA_GIZMO,
    };
    struct MeshData {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        std::vector<glm::vec4> cameraGizmoVertices;

        MeshData() = default;

        explicit MeshData(MeshDataType dataType, std::filesystem::path fp);

    private:
        std::filesystem::path m_filePath;
        void meshFromObjFile();
        void meshFromPointCloud();
        void meshFromPlyFile();

        void meshFromCameraGizmo();
    };

    struct MeshComponent {
        MeshComponent() = default;

        explicit MeshComponent(std::filesystem::path path, MeshDataType meshDataType) : m_meshPath(std::move(path)), m_type(meshDataType) {
        };
        std::filesystem::path m_meshPath; // Path to the mesh file (e.g., OBJ, PLY)
        MeshDataType m_type;
        VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;

        float focalPoint = 1.0f;
    };

    struct MeshInstance {
        std::unique_ptr<Buffer> vertexBuffer;
        std::unique_ptr<Buffer> indexBuffer;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        // Additional data like vertex layout, primitive type, etc.
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        MeshDataType m_type;

        // Other GPU-specific resources
    };

    /*
    struct MeshComponent {

        MeshComponent() {
            loadModel(Utils::getModelsPath() / "obj" / "quad.obj");
            m_modelPath = Utils::getModelsPath() / "obj" / "quad.obj";
        }

        MeshComponent(const MeshComponent &) = delete;

        MeshComponent &operator=(const MeshComponent &other) {
            return *this;
        }

        explicit MeshComponent(const std::filesystem::path &modelPath) {
            loadModel(modelPath);
            loadTexture(modelPath);
            m_modelPath = modelPath;
        }

        explicit MeshComponent(uint32_t type) {
            loadCameraModelMesh();
            m_modelPath = "Generated";
        }

        Texture2D objTexture; // TODO Possibly make more empty textures to match our triple buffering?
        UBOCamera &getCameraModelMesh() { return m_cameraModelVertices; }

        bool usesUBOMesh() { return m_isCameraModelMesh; }


        const UUID &getUUID() const { return m_meshUUID; }

        void loadOBJ(std::filesystem::path modelPath);

        std::filesystem::path getModelPath() { return m_modelPath; }

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
        UUID m_meshUUID;

    };
*/
};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT
