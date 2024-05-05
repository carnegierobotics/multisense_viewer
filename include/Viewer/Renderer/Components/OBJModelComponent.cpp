//
// Created by magnus on 4/11/24.
//

#include "Viewer/Renderer/Components/OBJModelComponent.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Tools/Utils.h"

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


    void OBJModelComponent::loadModel(std::filesystem::path modelPath) {


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

                vertex.uv0 = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                auto [it, inserted] = uniqueVertices.try_emplace(vertex, m_vertices.size());
                if (inserted) {
                    m_vertices.push_back(vertex);
                }

                m_indices.push_back(it->second);
            }
        }

    }

    void OBJModelComponent::loadTexture(std::filesystem::path texturePath) {
        int texWidth = 0, texHeight = 0, texChannels = 0;
        auto path = texturePath.replace_extension(".png");
        m_pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!m_pixels) {
            Log::Logger::getInstance()->error("Failed to load texture: {}", texturePath.string());

        }
        m_texSize = 4 * texHeight * texWidth;
        m_texHeight = texHeight;
        m_texWidth = texWidth;


    }

};