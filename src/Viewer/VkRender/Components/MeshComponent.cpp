//
// Created by magnus on 4/11/24.
//

#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "tinyply.h"

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
    MeshData::MeshData(MeshDataType dataType) {
    }

    MeshData::MeshData(MeshDataType dataType, const std::filesystem::path &filePath) : m_filePath(filePath){
        switch (dataType) {

            case OBJ_FILE:
                meshFromObjFile();
                break;
            case POINT_CLOUD:
                meshFromPointCloud();
                break;
            case PLY_FILE:
                meshFromPlyFile();
                break;
        }
    }

    void MeshData::meshFromObjFile() {
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;


        if (!reader.ParseFromFile(m_filePath.string(), reader_config)) {
            if (!reader.Error().empty()) {
                //std::cerr << "TinyObjReader: " << reader.Error();
                Log::Logger::getInstance()->error("Failed to load .OBJ file {}", m_filePath.string());
            }
            return;
        }


        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ file empty {}", m_filePath.string());
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
        uniqueVertices.reserve(estimatedVertexCount); // Reserve space for unique vertices

        vertices.reserve(estimatedVertexCount); // Reserve space to avoid resizing
        indices.reserve(estimatedIndexCount); // Reserve space to avoid resizing

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

    void MeshData::meshFromPointCloud() {
        for (int y = 0; y < 600; ++y) {
            for (int x = 0; x < 960; ++x) {
                float u = static_cast<float>(x) / 960.0f;
                float v = static_cast<float>(y) / 600.0f;
                Vertex vertex{};
                vertex.uv0 = glm::vec2{u, v};
                vertices.push_back(vertex);
            }
        }
    }

    void MeshData::meshFromPlyFile() {
        std::ifstream ss(m_filePath.string(), std::ios::binary);
        if (ss.fail()) {
            throw std::runtime_error("Failed to open " + m_filePath.string());
        }

        tinyply::PlyFile file;
        file.parse_header(ss);

        std::shared_ptr<tinyply::PlyData> verticesData, facesData;

        try {
            verticesData = file.request_properties_from_element("vertex", {"x", "y", "z"});
        } catch (const std::exception &e) {
            throw std::runtime_error("tinyply exception: " + std::string(e.what()));
        }

        try {
            facesData = file.request_properties_from_element("face", {"vertex_indices"});
        } catch (const std::exception &e) {
            throw std::runtime_error("tinyply exception: " + std::string(e.what()));
        }

        file.read(ss);

        const size_t numVertices = verticesData->count;
        const size_t numFaces = facesData->count;

        std::vector<float> verts(numVertices * 5);
        std::memcpy(verts.data(), verticesData->buffer.get(), verticesData->buffer.size_bytes());

        std::vector<uint32_t> faces(numFaces * 3);
        std::memcpy(faces.data(), facesData->buffer.get(), facesData->buffer.size_bytes());

        // Populate vertices
        for (size_t i = 0; i < numVertices; ++i) {
            Vertex vertex{};
            vertex.pos = {
                    verts[3 * i + 0],
                    verts[3 * i + 1],
                    verts[3 * i + 2]
            };
            vertex.uv0 = {
                    verts[3 * i + 3],
                    verts[3 * i + 4]
            };

            vertex.normal = glm::vec3(0.0f);
            vertices.push_back(vertex);
        }

        // Populate indices
        for (size_t i = 0; i < numFaces; ++i) {
            indices.push_back(faces[3 * i + 0]);
            indices.push_back(faces[3 * i + 1]);
            indices.push_back(faces[3 * i + 2]);
        }

        // Iterate through each triangle (indices.size() should be divisible by 3)
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i + 0];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            // Get the three vertices of the triangle
            glm::vec3 v0 = vertices[i0].pos;
            glm::vec3 v1 = vertices[i1].pos;
            glm::vec3 v2 = vertices[i2].pos;

            // Compute the two edges of the triangle
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;

            // Calculate the normal using cross product of edge1 and edge2
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

            // Accumulate the face normal into each vertex normal
            vertices[i0].normal += faceNormal;
            vertices[i1].normal += faceNormal;
            vertices[i2].normal += faceNormal;
        }

// Normalize all the vertex normals
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i].normal = glm::normalize(vertices[i].normal);
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
