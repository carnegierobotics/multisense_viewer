//
// Created by magnus on 11/27/24.
//


#include "Viewer/Rendering/MeshData.h"

#include "tinyply.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#define TINYOBJLOADER_USE_MAPBOX_EARCUT

#include <multisense_viewer/external/tinyobjloader/tiny_obj_loader.h>
#include <stb_image.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/hash.hpp>
#include <utility>

#include "IMeshParameters.h"


namespace VkRender {

    void MeshData::generateCylinderMesh(const CylinderMeshParameters& parameters) {
        // Extract parameters from the map
        glm::vec3 origin = parameters.origin;
        glm::vec3 direction =  parameters.direction;
        float magnitude =  parameters.magnitude;

        // Parameters for the cylinder
        const int segments = 20; // Adjust for smoother cylinder
        const float radius = parameters.radius; // Adjust as needed
        const float height = magnitude;

        glm::vec3 endPoint = origin + direction * height;

        // Generate the cylinder vertices and indices
        // Define the base circle and top circle vertices
        std::vector<Vertex> baseCircleVertices;
        std::vector<Vertex> topCircleVertices;

        for (int i = 0; i < segments; ++i) {
            float theta = 2.0f * glm::pi<float>() * float(i) / float(segments);
            float x = radius * cos(theta);
            float z = radius * sin(theta);

            glm::vec3 offset = glm::vec3(x, 0.0f, z);

            // Rotate offset to align with the direction vector
            glm::vec3 defaultUp = glm::vec3(0.0f, 1.0f, 0.0f);
            glm::quat rotationQuat = glm::rotation(defaultUp, glm::normalize(direction));

            // Rotate the offset
            offset = rotationQuat * offset;

            // Base vertex
            Vertex baseVertex{};
            baseVertex.pos = origin + offset;
            baseVertex.normal = -direction;
            baseVertex.uv0 = glm::vec2(float(i) / segments, 0.0f);
            baseCircleVertices.push_back(baseVertex);

            // Top vertex
            Vertex topVertex{};
            topVertex.pos = endPoint + offset;
            topVertex.normal = direction;
            topVertex.uv0 = glm::vec2(float(i) / segments, 1.0f);
            topCircleVertices.push_back(topVertex);
        }

        // Combine vertices
        vertices.reserve(segments * 2);
        vertices.insert(vertices.end(), baseCircleVertices.begin(), baseCircleVertices.end());
        vertices.insert(vertices.end(), topCircleVertices.begin(), topCircleVertices.end());

        // Generate indices for the side faces
        for (int i = 0; i < segments; ++i) {
            int next = (i + 1) % segments;
            int baseIndex = i;
            int topIndex = i + segments;
            int nextBaseIndex = next;
            int nextTopIndex = next + segments;

            // First triangle of quad
            indices.push_back(baseIndex);
            indices.push_back(nextBaseIndex);
            indices.push_back(topIndex);

            // Second triangle of quad
            indices.push_back(nextBaseIndex);
            indices.push_back(nextTopIndex);
            indices.push_back(topIndex);
        }

        // Generate indices for the base and top caps if desired
        // Base cap

        for (int i = 1; i < segments - 1; ++i) {
            indices.push_back(0);
            indices.push_back(i);
            indices.push_back(i + 1);
        }

        // Top cap
        for (int i = 1; i < segments - 1; ++i) {
            indices.push_back(segments);
            indices.push_back(segments + i + 1);
            indices.push_back(segments + i);
        }


        isDynamic = true;
    }


    void MeshData::generateCameraGizmoMesh(const CameraGizmoMeshParameters& params) {
        float a = 0.25f;
        float h = params.focalPoint;

        std::vector<glm::vec3> uboVertices = {
            // Pyramid base + apex
            glm::vec3(-a, -a, 0.0f), // 0: A
            glm::vec3( a, -a, 0.0f), // 1: B
            glm::vec3( a,  a, 0.0f), // 2: C
            glm::vec3(-a,  a, 0.0f), // 3: D
            glm::vec3( 0.0f, 0.0f, h), // 4: E (apex)

            // Top indicator vertices
            glm::vec3(-0.4f, 0.6f, 0.0f), // 5: F
            glm::vec3( 0.4f, 0.6f, 0.0f), // 6: G
            glm::vec3( 0.0f, 1.0f, 0.0f)  // 7: H
        };
        indices = {
            // Base
            3, 0, 1,
            3, 1, 2,

            // Sides
            0, 4, 1,
            1, 4, 2,
            2, 4, 3,
            3, 4, 0,

            // Top indicator
            5, 6, 7
        };

        vertices.resize(uboVertices.size());
        for (size_t i = 0; i < uboVertices.size(); ++i) {
            vertices[i].pos = uboVertices[i];
            vertices[i].normal   = glm::vec3(0.0f, 0.0f, 1.0f); // Placeholder normal
            vertices[i].uv0      = glm::vec2(0.0f, 0.0f);       // Placeholder UV
            vertices[i].uv1      = glm::vec2(0.0f, 0.0f);       // Placeholder UV
            vertices[i].color    = glm::vec4(1.0f);             // White color
        }

        isDynamic = true;
    }

    void MeshData::generateOBJMesh(const OBJFileMeshParameters& parameters) {
                tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;


        if (!reader.ParseFromFile(parameters.path.string(), reader_config)) {
            if (!reader.Error().empty()) {
                //std::cerr << "TinyObjReader: " << reader.Error();
                Log::Logger::getInstance()->error("Failed to load .OBJ file {}", parameters.path.string());
            }
            return;
        }


        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ file empty {}", parameters.path.string());
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        // Pre-allocate memory for vertices and indices vectors
        size_t estimatedVertexCount = attrib.vertices.size() / 3;
        size_t estimatedIndexCount = 0;
        for (const auto& shape : shapes) {
            estimatedIndexCount += shape.mesh.indices.size();
        }

        std::unordered_map<VkRender::Vertex, uint32_t> uniqueVertices{};
        uniqueVertices.reserve(estimatedVertexCount); // Reserve space for unique vertices

        vertices.reserve(estimatedVertexCount); // Reserve space to avoid resizing
        indices.reserve(estimatedIndexCount); // Reserve space to avoid resizing

        // Using range-based loop to process vertices and indices
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
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
                }
                else {
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

    void MeshData::generatePLYMesh(const PLYFileMeshParameters& parameters) {
                std::ifstream ss(parameters.path.string(), std::ios::binary);
        if (ss.fail()) {
            Log::Logger::getInstance()->warning("Failed to open {}", parameters.path.string());
            return;
        }

        tinyply::PlyFile file;
        file.parse_header(ss);

        std::shared_ptr<tinyply::PlyData> positionData, colorData, facesData;

        try {
            // Request position data (double type)
            positionData = file.request_properties_from_element("vertex", {"x", "y", "z"});
        }
        catch (const std::exception& e) {
            throw std::runtime_error("tinyply exception: " + std::string(e.what()));
        }

        try {
            // Request color data (uchar type)
            colorData = file.request_properties_from_element("vertex", {"red", "green", "blue"});
        }
        catch (const std::exception& e) {
            throw std::runtime_error("tinyply exception: " + std::string(e.what()));
        }

        try {
            facesData = file.request_properties_from_element("face", {"vertex_indices"});
        }
        catch (const std::exception& e) {
            throw std::runtime_error("tinyply exception: " + std::string(e.what()));
        }

        file.read(ss);

        const size_t numVertices = positionData->count;
        const size_t numFaces = facesData->count;

        // Load position data (double precision)
        std::vector<double> positions(numVertices * 3);
        std::memcpy(positions.data(), positionData->buffer.get(), positionData->buffer.size_bytes());

        // Load color data (unsigned char precision)
        std::vector<uint8_t> colors(numVertices * 3);
        std::memcpy(colors.data(), colorData->buffer.get(), colorData->buffer.size_bytes());

        std::vector<uint32_t> faces(numFaces * 3);
        std::memcpy(faces.data(), facesData->buffer.get(), facesData->buffer.size_bytes());

        // Populate vertices
        for (size_t i = 0; i < numVertices; ++i) {
            Vertex vertex{};
            vertex.pos = {
                static_cast<float>(positions[3 * i + 0]),
                static_cast<float>(positions[3 * i + 1]),
                static_cast<float>(positions[3 * i + 2])
            };

            // Convert color values from uint8 to float [0, 1]
            vertex.color = {
                colors[3 * i + 0] / 255.0f,
                colors[3 * i + 1] / 255.0f,
                colors[3 * i + 2] / 255.0f,
                1.0f // Alpha channel set to 1.0
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

        // Compute face normals and accumulate them in each vertex normal
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i + 0];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            glm::vec3 v0 = vertices[i0].pos;
            glm::vec3 v1 = vertices[i1].pos;
            glm::vec3 v2 = vertices[i2].pos;

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

            vertices[i0].normal += faceNormal;
            vertices[i1].normal += faceNormal;
            vertices[i2].normal += faceNormal;
        }

        // Normalize all the vertex normals
        for (auto& vertex : vertices) {
            vertex.normal = glm::normalize(vertex.normal);
        }

    }
}
