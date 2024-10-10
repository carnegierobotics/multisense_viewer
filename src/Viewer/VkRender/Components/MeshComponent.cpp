//
// Created by magnus on 4/11/24.
//

#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#define TINYOBJLOADER_USE_MAPBOX_EARCUT

#include <tiny_obj_loader.h>
#include <stb_image.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/hash.hpp>

namespace std {
    template<>
    struct hash<VkRender::Vertex> {
        size_t operator()(VkRender::Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                     (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.uv0) << 1);
        }
    };
};

namespace VkRender {

    MeshData::MeshData(const std::filesystem::path &filePath) {

        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;


        if (!reader.ParseFromFile(filePath.string(), reader_config)) {
            if (!reader.Error().empty()) {
                //std::cerr << "TinyObjReader: " << reader.Error();
                Log::Logger::getInstance()->error("Failed to load .OBJ file {}", filePath.string());
            }
            return;
        }


        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ file empty {}", filePath.string());
        }

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();
        auto &materials = reader.GetMaterials();

// Pre-allocate memory for vertices and indices vectors
        size_t estimatedVertexCount = attrib.vertices.size() / 3;
        size_t estimatedIndexCount = 0;
        for (const auto &shape: shapes) {
            estimatedIndexCount += shape.mesh.indices.size();
        }

        std::unordered_map<VkRender::Vertex, uint32_t> uniqueVertices{};
        uniqueVertices.reserve(estimatedVertexCount);  // Reserve space for unique vertices

        vertices.reserve(estimatedVertexCount);  // Reserve space to avoid resizing
        indices.reserve(estimatedIndexCount);     // Reserve space to avoid resizing

        // Using range-based loop to process vertices and indices
        for (const auto &shape: shapes) {
            for (const auto &index: shape.mesh.indices) {
                Vertex vertex{};
                vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                };

                if (index.normal_index > -1) {
                    vertex.normal = {
                            attrib.normals[3 * index.normal_index + 0],
                            attrib.normals[3 * index.normal_index + 1],
                            attrib.normals[3 * index.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 0.0f, 1.0f}; // Default normal if not present
                }

                if (index.texcoord_index > -1) {
                    vertex.uv0 = {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                }
                auto [it, inserted] = uniqueVertices.try_emplace(vertex, vertices.size());
                if (inserted) {
                    vertices.push_back(vertex);
                }

                indices.push_back(it->second);
            }
        }
    }

    /*
    void MeshComponent::loadModel(const std::filesystem::path &modelPath) {


        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;


        if (!reader.ParseFromFile(modelPath.string(), reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
                Log::Logger::getInstance()->error("Failed to load .OBJ file {}", modelPath.string());
            }
            return;
        }


        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ file empty {}", modelPath.string());
        }

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();
        auto &materials = reader.GetMaterials();

// Pre-allocate memory for vertices and indices vectors
        size_t estimatedVertexCount = attrib.vertices.size() / 3;
        size_t estimatedIndexCount = 0;
        for (const auto &shape: shapes) {
            estimatedIndexCount += shape.mesh.indices.size();
        }

        std::unordered_map<VkRender::Vertex, uint32_t> uniqueVertices{};
        uniqueVertices.reserve(estimatedVertexCount);  // Reserve space for unique vertices

        m_vertices.reserve(estimatedVertexCount);  // Reserve space to avoid resizing
        m_indices.reserve(estimatedIndexCount);     // Reserve space to avoid resizing

        // Using range-based loop to process vertices and indices
        for (const auto &shape: shapes) {
            for (const auto &index: shape.mesh.indices) {
                Vertex vertex{};
                vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                };

                if (index.normal_index > -1) {
                    vertex.normal = {
                            attrib.normals[3 * index.normal_index + 0],
                            attrib.normals[3 * index.normal_index + 1],
                            attrib.normals[3 * index.normal_index + 2]
                    };
                } else {
                    vertex.normal = {0.0f, 0.0f, 1.0f}; // Default normal if not present
                }

                if (index.texcoord_index > -1) {
                    vertex.uv0 = {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                }
                auto [it, inserted] = uniqueVertices.try_emplace(vertex, m_vertices.size());
                if (inserted) {
                    m_vertices.push_back(vertex);
                }

                m_indices.push_back(it->second);
            }
        }

    }

    void MeshComponent::loadTexture(const std::filesystem::path &texturePath) {
        int texWidth = 0, texHeight = 0, texChannels = 0;
        std::filesystem::path texPath = texturePath;
        auto path = texPath.replace_extension(".png");
        m_pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!m_pixels) {
            Log::Logger::getInstance()->error("Failed to load texture: {}", texturePath.string());

        }
        m_texSize = STBI_rgb_alpha * texHeight * texWidth;
        m_texHeight = texHeight;
        m_texWidth = texWidth;
        // TODO free m_pixels once we are done with it. We are currently leaving memory leaks


    }

    void MeshComponent::loadCameraModelMesh() {


        float a = 0.5;
        float h = 2.0;
        m_cameraModelVertices.positions = {
                // Base (CCW from top)
                glm::vec4(-a, a, 0, 1.0), // D
                glm::vec4(-a, -a, 0, 1.0), // A
                glm::vec4(a, -a, 0, 1.0), // B
                glm::vec4(-a, a, 0, 1.0), // D
                glm::vec4(a, -a, 0, 1.0), // B
                glm::vec4(a, a, 0, 1.0), // C

                // Side 1
                glm::vec4(-a, -a, 0, 1.0), // A
                glm::vec4(0, 0, h, 1.0), // E
                glm::vec4(a, -a, 0, 1.0f), // B

                // Side 2
                glm::vec4(a, -a, 0, 1.0), // B
                glm::vec4(0, 0, h, 1.0), // E
                glm::vec4(a, a, 0, 1.0), // C

                // Side 3
                glm::vec4(a, a, 0, 1.0), // C
                glm::vec4(0, 0, h, 1.0), // E
                glm::vec4(-a, a, 0, 1.0), // D

                // Side 4
                glm::vec4(-a, a, 0, 1.0), // D
                glm::vec4(0, 0, h, 1.0), // E
                glm::vec4(-a, -a, 0, 1.0), // A

                // Top indicator
                glm::vec4(-0.4, 0.6, 0, 1.0), // D
                glm::vec4(0.4, 0.6, 0, 1.0), // E
                glm::vec4(0, 1.0, 0, 1.0) // A

        };

        m_isCameraModelMesh = true;
    }

    void MeshComponent::loadOBJ(std::filesystem::path modelPath) {
        m_vertices.clear();
        m_indices.clear();

        loadModel(modelPath);
        loadTexture(modelPath);
        m_modelPath = modelPath;
        m_meshUUID = UUID(); // Generate new UUID
    }

    */

};