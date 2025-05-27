//
// Created by magnus on 11/27/24.
//

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/ext.hpp"


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include <tiny_obj_loader.h>
#include <stb_image.h>


#include <utility>
#include "MeshParameters.h"
#include "Viewer/Rendering/MeshData.h"




namespace VkRender {
    static glm::vec3 getViridisColor(float t) {
        // Clamp t to [0, 1]
        t = glm::clamp(t, 0.0f, 1.0f);

        // Viridis control points (from known values)
        struct ViridisPoint {
            float t;
            glm::vec3 color;
        };
        static const ViridisPoint viridis[5] = {
            {0.0f, glm::vec3(0.267004f, 0.004874f, 0.329415f)},
            {0.25f, glm::vec3(0.229739f, 0.322361f, 0.545706f)},
            {0.5f, glm::vec3(0.127568f, 0.566949f, 0.550556f)},
            {0.75f, glm::vec3(0.369214f, 0.788888f, 0.382914f)},
            {1.0f, glm::vec3(0.993248f, 0.906157f, 0.143936f)}
        };

        // Find the interval [viridis[i].t, viridis[i+1].t] that contains t.
        for (int i = 0; i < 4; ++i) {
            if (t <= viridis[i + 1].t) {
                float localT = (t - viridis[i].t) / (viridis[i + 1].t - viridis[i].t);
                return glm::mix(viridis[i].color, viridis[i + 1].color, localT);
            }
        }
        return viridis[4].color;
    }
    void MeshData::generateGaussian2DMesh(const Gaussian2DMeshParameters &params) {
        const int   N       = 50;
        const float σx      = params.covX;
        const float σy      = params.covY;
        const float thresh  = params.threshold;

        // span ±3σ in each axis
        const glm::vec2 min = { -3.0f*σx, -3.0f*σy };
        const glm::vec2 max = { +3.0f*σx, +3.0f*σy };

        const float dx = (max.x - min.x) / float(N - 1);
        const float dy = (max.y - min.y) / float(N - 1);

        std::vector<int>     vertexMap(N * N, -1);
        std::vector<Vertex>  tmpVertices;  tmpVertices.reserve(N * N);
        std::vector<uint32_t>tmpIndices;   tmpIndices.reserve(2 * (N - 1) * (N - 1) * 3);

        // Anisotropic Gaussian kernel
        auto gaussianKernel = [&](float x, float y) {
            float r2 = (x*x)/(σx*σx) + (y*y)/(σy*σy);
            return std::exp(-0.5f * r2);
        };

        for (int i = 0; i < N; ++i) {
            float x = min.x + i*dx;
            for (int j = 0; j < N; ++j) {
                float y = min.y + j*dy;

                float g = gaussianKernel(x, y);
                if (g < thresh) {
                    vertexMap[i*N + j] = -1;
                    continue;
                }

                Vertex v{};
                v.pos    = { x, y, 0.0f };
                v.normal = { 0.0f, 0.0f, 1.0f };
                // color RGB fixed; opacity scaled by g
                v.color  = glm::vec4(params.color, g * params.opacity);

                vertexMap[i*N + j] = int(tmpVertices.size());
                tmpVertices.push_back(v);
            }
        }

        // Create two triangles per cell
        for (int i = 0; i < N - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                int v0 = vertexMap[i * N + j];
                int v1 = vertexMap[i * N + (j + 1)];
                int v2 = vertexMap[(i + 1) * N + j];
                int v3 = vertexMap[(i + 1) * N + (j + 1)];

                // If valid, create two triangles
                if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                    m_indices.push_back(v0);
                    m_indices.push_back(v1);
                    m_indices.push_back(v2);

                    m_indices.push_back(v1);
                    m_indices.push_back(v3);
                    m_indices.push_back(v2);
                }
            }
        }

        // Finalize
        m_vertices = std::move(tmpVertices);

    }

    void MeshData::generateQuadricMesh(const QuadricMeshParameters &params) {
        int N = params.gridResolution;
        float dx = (params.max.x - params.min.x) / float(N - 1);
        float dy = (params.max.y - params.min.y) / float(N - 1);
        // Precompute the sign factors alpha_x, alpha_y
        float alphaX = std::tanh(params.t_x);
        float alphaY = std::tanh(params.t_y);
        // We'll store a "map" of valid vertex m_indices. -1 means not used.
        std::vector<int> vertexMap(N * N, -1);
        std::vector<Vertex> tmpVertices;
        tmpVertices.reserve(N * N);
        // If you want a final global scale (e.g. 0.1), define here
        float scaleFactor = 1.0f;
        // Helper lambda for Beta kernel:
        auto betaKernel = [&](float r, float bExp) {
            // (1 - r^2)^(4 e^bExp), clipped if r>1
            if (r > 1.0f) r = 1.0f;
            return std::pow(1.0f - r * r, 4.0f * std::exp(bExp));
        };
        // Generate grid and compute vertices
        for (int i = 0; i < N; ++i) {
            float x = params.min.x + i * dx; // domain from min.x to max.x
            for (int j = 0; j < N; ++j) {
                float y = params.min.y + j * dy; // domain from min.y to max.y
                float z = 0;
                glm::vec3 position(x, y, z);

                //float zSquared = params.a * params.a *(x*x + y * y);
                //float z = sqrtf(zSquared);
                // Build position
                // Evaluate gradient for normal
                glm::vec3 grad(
                    2.0f * params.c * alphaX * x / (params.a * params.a),
                    2.0f * params.c * alphaY * y / (params.b * params.b),
                    -1.0f
                );
                glm::vec3 normal = glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f));
                // Apply scale factor if you like
                position *= scaleFactor;

                float geodesic = sqrtf(std::pow(x, 2.0f) + std::pow(y, 2.0f) + std::pow(z, 2.0f)) / params.kernelScale;
                // Evaluate kernel
                float bkValue = betaKernel(geodesic, params.b_beta);

                /*
                if (!keepVertex) {
                    vertexMap[i * N + j] = -1;
                    continue;
                }
                */

                float opacity = (bkValue > params.threshold) ? bkValue : 0.0f;

                // Construct vertex
                Vertex v{};
                v.color = glm::vec4(getViridisColor(bkValue), opacity);
                v.pos = position;
                v.normal = normal;

                int newIndex = static_cast<int>(tmpVertices.size());
                vertexMap[i * N + j] = newIndex;
                tmpVertices.push_back(v);


                /*
                float rho = sqrtf(std::pow(x, 2.0f) + std::pow(y, 2.0f));
                float theta = atan2f(y, x);

                // Compute a(theta) = c * ( (alphaX * cos²(theta))/(a²) + (alphaY * sin²(theta))/(b²) )
                float a_theta = params.c * ((alphaX * std::cos(theta) * std::cos(theta)) / (params.a * params.a)
                                     + (alphaY * std::sin(theta) * std::sin(theta)) / (params.b * params.b));

                const float epsilon = std::numeric_limits<float>::epsilon();

                // Compute the geodesic distance l(ρ) along the surface
                float geodesicDistCircular = 0.0f;
                if (std::fabs(a_theta) > epsilon) {
                    // l(ρ) = (ρ/2)*sqrt(1+4a(θ)²ρ²) + asinh(2a(θ)ρ)/(4a(θ))
                    float term1 = 0.5f * rho * std::sqrt(1.0f + 4.0f * a_theta * a_theta * rho * rho);
                    float term2 = std::asinh(2.0f * a_theta * rho) / (4.0f * a_theta);
                    geodesicDistCircular = term1 + term2;
                } else {
                    // When a(θ) is nearly zero, use Euclidean distance.
                    geodesicDistCircular = rho;
                }



                // Compute the geodesic distances separately along x and y
                float rho_x = std::fabs(x);
                float rho_y = std::fabs(y);

                float geodesic_x = 0.0f;
                float geodesic_y = 0.0f;

                if (std::fabs(a_theta) > epsilon) {
                    auto computeGeodesic = [&](float rho_val) -> float {
                        float term1 = 0.5f * rho_val * std::sqrt(1.0f + 4.0f * a_theta * a_theta * rho_val * rho_val);
                        float term2 = std::asinh(2.0f * a_theta * rho_val) / (4.0f * a_theta);
                        return term1 + term2;
                    };

                    geodesic_x = computeGeodesic(rho_x);
                    geodesic_y = computeGeodesic(rho_y);
                } else {
                    // When a(θ) is nearly zero, use Euclidean distance.
                    geodesic_x = rho_x;
                    geodesic_y = rho_y;
                }

                // Use max norm to enforce square-like level sets
                float geodesicDistSquare = std::max(geodesic_x, geodesic_y);

                float geodesicDist = glm::mix(geodesicDistSquare, geodesicDistCircular, params.circularity);
                // Normalize the geodesic distance by kernelScale
                float r = geodesicDist / params.kernelScale;

*/
            }
        }

        // Create two triangles per cell
        for (int i = 0; i < N - 1; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                int v0 = vertexMap[i * N + j];
                int v1 = vertexMap[i * N + (j + 1)];
                int v2 = vertexMap[(i + 1) * N + j];
                int v3 = vertexMap[(i + 1) * N + (j + 1)];

                // If valid, create two triangles
                if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                    m_indices.push_back(v0);
                    m_indices.push_back(v1);
                    m_indices.push_back(v2);

                    m_indices.push_back(v1);
                    m_indices.push_back(v3);
                    m_indices.push_back(v2);
                }
            }
        }

        // Finalize
        m_vertices = std::move(tmpVertices);
    }


    void MeshData::generateCylinderMesh(const CylinderMeshParameters &parameters) {
        // Extract parameters from the map
        glm::vec3 origin = parameters.origin;
        glm::vec3 direction = glm::normalize(parameters.direction);
        float magnitude = parameters.magnitude;

        // Parameters for the cylinder
        const int segments = 20; // Adjust for smoother cylinder
        const float radius = parameters.radius; // Adjust as needed
        const float height = magnitude;

        glm::vec3 endPoint = origin + direction * height;

        // Generate the cylinder vertices and m_indices
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
            baseVertex.pos = glm::vec4(origin + offset, 0.0f);
            baseCircleVertices.push_back(baseVertex);

            // Top vertex
            Vertex topVertex{};
            topVertex.pos = glm::vec4(endPoint + offset, 0.0f);
            topCircleVertices.push_back(topVertex);
        }

        // Combine vertices
        m_vertices.reserve(segments * 2);
        m_vertices.insert(m_vertices.end(), baseCircleVertices.begin(), baseCircleVertices.end());
        m_vertices.insert(m_vertices.end(), topCircleVertices.begin(), topCircleVertices.end());

        // Generate m_indices for the side faces
        for (int i = 0; i < segments; ++i) {
            int next = (i + 1) % segments;
            int baseIndex = i;
            int topIndex = i + segments;
            int nextBaseIndex = next;
            int nextTopIndex = next + segments;

            // First triangle of quad
            m_indices.push_back(baseIndex);
            m_indices.push_back(nextBaseIndex);
            m_indices.push_back(topIndex);

            // Second triangle of quad
            m_indices.push_back(nextBaseIndex);
            m_indices.push_back(nextTopIndex);
            m_indices.push_back(topIndex);
        }

        // Generate m_indices for the base and top caps if desired
        // Base cap

        for (int i = 1; i < segments - 1; ++i) {
            m_indices.push_back(0);
            m_indices.push_back(i);
            m_indices.push_back(i + 1);
        }

        // Top cap
        for (int i = 1; i < segments - 1; ++i) {
            m_indices.push_back(segments);
            m_indices.push_back(segments + i + 1);
            m_indices.push_back(segments + i);
        }

    }


    void MeshData::generateCameraPinholeGizmoMesh(const CameraGizmoPinholeMeshParameters &pinhole) {
        float width = pinhole.parameters.width;
        float height = pinhole.parameters.height;
        float fx = pinhole.parameters.fx;
        float fy = pinhole.parameters.fy;
        float cx = pinhole.parameters.cx;
        float cy = pinhole.parameters.cy;
        float focalLength = pinhole.parameters.focalLength;

        // Choose a plane at Z = -1 for visualization. Objects in front of the camera have negative Z.

        // Convert image corners into camera coordinates:
        // We'll map four corners of the image:
        // Top-left:     (u,v) = (0, 0)
        // Top-right:    (u,v) = (width, 0)
        // Bottom-right: (u,v) = (width, height)
        // Bottom-left:  (u,v) = (0, height)
        //
        // The mapping chosen so that:
        // X = -(u - cx)*Z_plane/fx
        // Y =  (v - cy)*Z_plane/fy
        // Z =  Z_plane (negative)
        //
        // This ensures:
        // - Left side (u<cx) => negative X
        // - Top side (v<cy) => positive Y
        // - Forward along negative Z

        auto mapPixelTo3D = [&](float u, float v, float Z_plane) {
            float X = -(u - cx) * Z_plane / fx;
            float Y = -(v - cy) * Z_plane / fy; // Notice the minus sign before (v - cy)
            float Z = Z_plane;
            return glm::vec3(X, Y, Z);
        };

        float Z_plane = -1.0f;

        glm::vec3 frontTopLeft = mapPixelTo3D(0.0f, 0.0f, Z_plane);
        glm::vec3 frontTopRight = mapPixelTo3D(width, 0.0f, Z_plane);
        glm::vec3 frontBottomRight = mapPixelTo3D(width, height, Z_plane);
        glm::vec3 frontBottomLeft = mapPixelTo3D(0.0f, height, Z_plane);

        glm::vec3 backTopLeft = mapPixelTo3D(0.0f, 0.0f, focalLength / 1000); // convert focal length to meters
        glm::vec3 backTopRight = mapPixelTo3D(width, 0.0f, focalLength / 1000); // convert focal length to meters
        glm::vec3 backBottomRight = mapPixelTo3D(width, height, focalLength / 1000); // convert focal length to meters
        glm::vec3 backBottomLeft = mapPixelTo3D(0.0f, height, focalLength / 1000); // convert focal length to meters

        // The pinhole (camera center) at the origin
        glm::vec3 pinholePos = glm::vec3(0.0f, 0.0f, 0.0f);

        // Define vertices: pinhole + the four corners of the image plane
        // We'll store them in order: pinhole(0), topLeft(1), topRight(2), bottomRight(3), bottomLeft(4)
        std::vector<glm::vec3> uboVertices = {
            pinholePos,
            frontTopLeft,
            frontTopRight,
            frontBottomRight,
            frontBottomLeft,
            backTopLeft,
            backTopRight,
            backBottomRight,
            backBottomLeft
        };

        // We'll create a simple line-based mesh:
        // Lines from pinhole to each corner:
        // pinhole -> topLeft
        // pinhole -> topRight
        // pinhole -> bottomRight
        // pinhole -> bottomLeft
        //
        // And lines forming the image plane rectangle:
        // topLeft -> topRight
        // topRight -> bottomRight
        // bottomRight -> bottomLeft
        // bottomLeft -> topLeft

    m_indices = {
        // pyramid sides
        0, 2, 1,
        0, 3, 2,
        0, 4, 3,
        0, 1, 4,

        // front image‐plane quad
        1, 2, 3,
        1, 3, 4,

        // back sensor‐plane quad
        5, 7, 8,
        5, 6, 7
    };


        m_vertices.resize(uboVertices.size());
        for (size_t i = 0; i < uboVertices.size(); ++i) {
            m_vertices[i].pos = glm::vec4(uboVertices[i], 0.0f);
            m_vertices[i].color = glm::vec4(1.0f); // White color
        }

        // This is a gizmo; often drawn as lines. Ensure rendering mode is line-friendly if needed.
    }

    void MeshData::generateCameraPerspectiveGizmoMesh(const CameraGizmoPerspectiveMeshParameters &perspective) {
        float nearDist = perspective.parameters.nearPlane;
        float farDist = perspective.parameters.farPlane;
        float fovDegrees = perspective.parameters.fov;
        float aspect = perspective.parameters.aspect;

        float fovRadians = glm::radians(fovDegrees);
        float tanHalfFov = std::tan(fovRadians * 0.5f);

        float nearHeight = 2.0f * nearDist * tanHalfFov;
        float nearWidth = nearHeight * aspect;

        float farHeight = 2.0f * farDist * tanHalfFov;
        float farWidth = farHeight * aspect;

        // Near‐plane corners (z = –nearDist)
        glm::vec3 NBL(-nearWidth * 0.5f, -nearHeight * 0.5f, -nearDist);
        glm::vec3 NBR(nearWidth * 0.5f, -nearHeight * 0.5f, -nearDist);
        glm::vec3 NTR(nearWidth * 0.5f, nearHeight * 0.5f, -nearDist);
        glm::vec3 NTL(-nearWidth * 0.5f, nearHeight * 0.5f, -nearDist);

        // Far‐plane corners (z = –farDist)
        glm::vec3 FBL(-farWidth * 0.5f, -farHeight * 0.5f, -farDist);
        glm::vec3 FBR(farWidth * 0.5f, -farHeight * 0.5f, -farDist);
        glm::vec3 FTR(farWidth * 0.5f, farHeight * 0.5f, -farDist);
        glm::vec3 FTL(-farWidth * 0.5f, farHeight * 0.5f, -farDist);

        std::vector<glm::vec3> uboVertices = {
            NBL, NBR, NTR, NTL, // 0–3 near
            FBL, FBR, FTR, FTL // 4–7 far
        };

        // Flip every triangle (a,b,c) → (a,c,b)
        m_indices = {
            // Near face
            0, 2, 1,
            0, 3, 2,

            // Far face
            4, 6, 5,
            4, 7, 6,

            // Left face
            0, 7, 3,
            0, 4, 7,

            // Right face
            1, 6, 2,
            1, 5, 6,

            // Top face (NTL, NTR, FTR, FTL)
            3, 2, 6,
            6, 7, 3,

            // Bottom face (NBL, NBR, FBR, FBL)
            0, 1, 5,
            5, 4, 0
        };

        m_vertices.resize(uboVertices.size());
        for (size_t i = 0; i < uboVertices.size(); ++i) {
            m_vertices[i].pos = glm::vec4(uboVertices[i], 0.0f);
            m_vertices[i].color = glm::vec4(1.0f);
        }
    }

    void MeshData::generateOBJMesh(const OBJFileMeshParameters &parameters) {
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./";
        reader_config.triangulate = false; // we’ll do it manually

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(parameters.path.string(), reader_config)) {
            Log::Logger::getInstance()->error("Failed to load .OBJ file {}", parameters.path.string());
            return;
        }
        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ warning: {}", reader.Warning());
        }

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();

        bool hasNormals = !attrib.normals.empty();
        bool hasTexcoords = !attrib.texcoords.empty();

        // Estimate sizes
        size_t estVerts = attrib.vertices.size() / 3;
        size_t estIdxs = 0;
        for (auto &shape: shapes)
            for (auto vcount: shape.mesh.num_face_vertices)
                estIdxs += (vcount - 2) * 3;

        m_vertices.clear();
        m_indices.clear();
        m_vertices.reserve(estVerts);
        m_indices.reserve(estIdxs);

        std::unordered_map<VkRender::Vertex, uint32_t> uniqueVerts;
        uniqueVerts.reserve(estVerts);

        // Helper lambda to add a single corner
        auto addCorner = [&](const tinyobj::index_t &idx) {
            VkRender::Vertex v{};
            // POSITION
            v.pos = {
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2]
            };
            // NORMAL (or zero)
            if (hasNormals && idx.normal_index >= 0) {
                v.normal = {
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]
                };
            } else {
                v.normal = {0.0f, 0.0f, 0.0f};
            }
            // UV (or zero)
            if (hasTexcoords && idx.texcoord_index >= 0) {
                v.uv0 = {
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
                };
            } else {
                v.uv0 = {0.0f, 0.0f};
            }
            // De-dup
            auto [it, inserted] = uniqueVerts.try_emplace(v, uint32_t(m_vertices.size()));
            if (inserted) {
                m_vertices.push_back(v);
            }
            return it->second;
        };

        // Build triangles
        for (auto &shape: shapes) {
            auto &mesh = shape.mesh;
            size_t offset = 0;
            for (size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
                int fv = mesh.num_face_vertices[f];
                // grab all the indices of this face
                std::vector<tinyobj::index_t> faceCorners;
                faceCorners.reserve(fv);
                for (int k = 0; k < fv; ++k) {
                    faceCorners.push_back(mesh.indices[offset + k]);
                }
                // fan-triangulate: (0, k, k+1)
                for (int k = 1; k + 1 < fv; ++k) {
                    m_indices.push_back(addCorner(faceCorners[0]));
                    m_indices.push_back(addCorner(faceCorners[k]));
                    m_indices.push_back(addCorner(faceCorners[k + 1]));
                }
                offset += fv;
            }
        }

        // if the OBJ had *no* normals, build them now
        if (!hasNormals) {
            computeNormals();
        }
    }

    void MeshData::computeNormals() {
        std::vector<glm::vec3> acc(m_vertices.size(), glm::vec3(0.0f));

        // accumulate face normals
        for (size_t i = 0; i + 2 < m_indices.size(); i += 3) {
            uint32_t i0 = m_indices[i + 0],
                    i1 = m_indices[i + 1],
                    i2 = m_indices[i + 2];
            auto &p0 = m_vertices[i0].pos;
            auto &p1 = m_vertices[i1].pos;
            auto &p2 = m_vertices[i2].pos;

            glm::vec3 fn = glm::normalize(glm::cross(p1 - p0, p2 - p0));
            acc[i0] += fn;
            acc[i1] += fn;
            acc[i2] += fn;
        }

        // normalize and assign only where we originally had no normal
        for (size_t i = 0; i < m_vertices.size(); ++i) {
            if (glm::length(m_vertices[i].normal) < 1e-6f) {
                m_vertices[i].normal = glm::normalize(acc[i]);
            }
        }
    }

void MeshData::generatePLYMesh(const PLYFileMeshParameters &p)
{
    // ----- 1. run Assimp -------------------------------------------------------
    constexpr uint32_t kFlags =
          aiProcess_Triangulate            // n‑gons → triangles
        | aiProcess_JoinIdenticalVertices  // merge duplicates, keeps index size down
        | aiProcess_GenSmoothNormals       // make normals if none in file
        | aiProcess_ValidateDataStructure; // catch malformed PLY early

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(p.path.string(), kFlags);

    if (!scene || !scene->HasMeshes()) {
        Log::Logger::getInstance()->warning("Assimp: failed to read {} ({})",
                     p.path.string(), importer.GetErrorString());
        return;
    }

    // ----- 2. flatten all meshes into a single VBO/IBO ------------------------
    m_vertices.clear();
    m_indices .clear();

    size_t baseVertex = 0;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
    {
        const aiMesh *mesh = scene->mMeshes[m];

        // 2‑a. vertices --------------------------------------------------------
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            Vertex v;
            // position (always present)
            const aiVector3D &p3 = mesh->mVertices[i];
            v.pos = { p3.x, p3.y, p3.z };

            // normal (generated above if missing in file)
            if (mesh->HasNormals()) {
                const aiVector3D &n3 = mesh->mNormals[i];
                v.normal = { n3.x, n3.y, n3.z };
            } else {
                v.normal = {};
            }

            // colour (only first colour set; falls back to white)
            if (mesh->HasVertexColors(0)) {
                const aiColor4D &c4 = mesh->mColors[0][i];
                v.color = { c4.r, c4.g, c4.b, c4.a };
            } else {
                v.color = {1.f, 1.f, 1.f, 1.f};
            }

            m_vertices.push_back(v);
        }

        // 2‑b. indices ---------------------------------------------------------
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
        {
            const aiFace &face = mesh->mFaces[f];
            if (face.mNumIndices != 3) {
                // aiProcess_Triangulate guarantees triangles, but be safe
                continue;
            }
            m_indices.push_back(baseVertex + face.mIndices[0]);
            m_indices.push_back(baseVertex + face.mIndices[1]);
            m_indices.push_back(baseVertex + face.mIndices[2]);
        }

        baseVertex += mesh->mNumVertices;
    }

    Log::Logger::getInstance()->info("PLY: loaded {} vertices, {} triangles from {}",
              m_vertices.size(), m_indices.size() / 3, p.path.string());
}


    void MeshData::generatePlaneMesh(const PlaneMeshParameters &plane) {
        // Origin in XYZ, and half-extents in X (width) and Y (height)
        const glm::vec3 O = plane.origin;
        const glm::vec2 half = glm::vec2(plane.size * 0.5f);

        // Four corners in the XY plane (Z fixed at O.z), CCW when looking down +Z
        const std::array<glm::vec3, 4> corners = {
            glm::vec3{O.x - half.x, O.y - half.y, O.z},
            glm::vec3{O.x + half.x, O.y - half.y, O.z},
            glm::vec3{O.x + half.x, O.y + half.y, O.z},
            glm::vec3{O.x - half.x, O.y + half.y, O.z}
        };

        // Up-vector normal for a Z-up world
        const glm::vec3 normal{0.0f, 0.0f, 1.0f};

        // matching “unit” UVs
        const std::array<glm::vec2,4> uvs = {{
            {0.0f, 0.0f},  // bottom-left
            {1.0f, 0.0f},  // bottom-right
            {1.0f, 1.0f},  // top-right
            {0.0f, 1.0f}   // top-left
        }};

        for (int i = 0; i < 4; ++i) {
            Vertex v{};
            v.pos    = corners[i];
            v.normal = normal;
            v.uv0    = uvs[i];
            m_vertices.push_back(v);
        }

        // Two triangles: (0,1,2) and (2,3,0)
        m_indices = {
            2, 1, 0,
            0, 3, 2
        };
    }

    void MeshData::generateCubeMesh(const CubeMeshParameters &cube) {
        m_vertices.clear();
        m_indices.clear();

        const glm::vec3 half = cube.getSize() / 2.0f;
        const glm::vec3 &O = cube.getOrigin();

        // same 8 corners
        const std::array<glm::vec3, 8> corners = {{
            {-half.x, -half.y, -half.z}, // 0
            { half.x, -half.y, -half.z}, // 1
            { half.x,  half.y, -half.z}, // 2
            {-half.x,  half.y, -half.z}, // 3
            {-half.x, -half.y,  half.z}, // 4
            { half.x, -half.y,  half.z}, // 5
            { half.x,  half.y,  half.z}, // 6
            {-half.x,  half.y,  half.z}  // 7
        }};

        // corrected normals: bottom/top on Z, left/right on X, back/front on Y
        const std::array<glm::vec3, 6> normals = {{
            { 0,  0, -1}, // 0 → bottom (Z−)
            { 0,  0,  1}, // 1 → top    (Z+)
            {-1,  0,  0}, // 2 → left   (X−)
            { 1,  0,  0}, // 3 → right  (X+)
            { 0, -1,  0}, // 4 → back   (Y−)
            { 0,  1,  0}  // 5 → front  (Y+)
        }};

        struct Face { uint8_t v[4], n; };
        // assign each quad the correct normal-index
        const std::array<Face, 6> faces = {{
            {{0, 1, 2, 3}, 0}, // bottom (Z−)
            {{4, 5, 6, 7}, 1}, // top    (Z+)
            {{4, 0, 3, 7}, 2}, // left   (X−)
            {{1, 5, 6, 2}, 3}, // right  (X+)
            {{0, 4, 5, 1}, 4}, // back   (Y−)
            {{3, 2, 6, 7}, 5}  // front  (Y+)
        }};

        const glm::vec4 white{1,1,1,1};

        for (int f = 0; f < 6; ++f) {
            const Face &face = faces[f];
            uint32_t base = static_cast<uint32_t>(m_vertices.size());

            for (int k = 0; k < 4; ++k) {
                Vertex v;
                v.pos    = O + corners[face.v[k]];
                v.normal = normals[face.n];
                m_vertices.push_back(v);
            }

            // CCW from outside
            m_indices.insert(m_indices.end(), {
                base+0, base+1, base+2,
                base+2, base+3, base+0
            });
        }
    }
}
