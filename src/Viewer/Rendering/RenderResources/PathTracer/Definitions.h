//
// Created by magnus-desktop on 12/8/24.
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <glm/glm.hpp>
#include <Viewer/Rendering/Components/TransformComponent.h>
#include <Viewer/Rendering/Editors/PinholeCamera.h>

#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Components/Components.h"

#ifdef DIFF_RENDERER_ENABLED
#include "torch/torch.h"
#endif

namespace VkRender::PathTracer {
#ifdef DIFF_RENDERER_ENABLED

    struct GPUDataTensors {
        torch::Tensor positions;
        torch::Tensor scales;
        torch::Tensor normals;

        // properties
        torch::Tensor emissions;
        torch::Tensor colors;
        torch::Tensor specular;
        torch::Tensor diffuse;

        // Quadrics:
        torch::Tensor quadrics; // nx12
        torch::Tensor quadricPositions; // 3 elems (pos)
        torch::Tensor quadricRotations; // 4 elems (Quat)
    };
#else
    struct GPUDataTensors;
#endif


    // Enum for kernel types
    typedef enum KernelType {
        KERNEL_PATH_TRACER_MESH,
        KERNEL_PATH_TRACER_2DGS,
        KERNEL_TYPE_COUNT // To count the number of kernels
    } KernelType;

    // Function to map KernelType to a string
    static const char *KernelTypeToString(KernelType kernel) {
        switch (kernel) {
            case KERNEL_PATH_TRACER_MESH: return "Path Tracer: Mesh";
            case KERNEL_PATH_TRACER_2DGS: return "Path Tracer: 2DGS";
            default: return "Unknown";
        }
    }

    // Function to map string to KernelType
    static KernelType StringToKernelType(const char *str) {
        if (strcmp(str, "Path Tracer: Mesh") == 0) return KERNEL_PATH_TRACER_MESH;
        if (strcmp(str, "Path Tracer: 2DGS") == 0) return KERNEL_PATH_TRACER_2DGS;
        return KERNEL_TYPE_COUNT; // Invalid
    }


    struct InputAssembly {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec3 normal;
    };

    struct GaussianInputAssembly {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 scale;

        float emission; // Emissive power
        glm::vec4 color; // Albedo
        float diffuse; // Diffuse coefficient
        float specular; // Specular coefficient
        float phongExponent; // Shininess exponent
    };

    struct QuadricInputAssembly {
        // Quadric parameters
        float a = 1.0f; // Scale in x
        float b = 1.0f; // Scale in y
        float c = 1.0f; // Curvature scale
        float t_x = 1.0f; // Param controlling sign in x-direction
        float t_y = 1.0f; // Param controlling sign in y-direction

        // Sampling parameters
        glm::vec2 min = glm::vec2(-10.0f);
        glm::vec2 max = glm::vec2(10.0f);

        // Beta kernel parameters:
        float b_beta = 0.0f; // exponent shift
        float threshold = 0.1f; // radial kernel threshold
        float kernelScale = 1.0f; // normalizes the radial coordinate

        // Appearance
        float emission; // Emissive power
        glm::vec4 color; // Albedo
        float diffuse; // Diffuse coefficient
        float specular; // Specular coefficient
        float phongExponent; // Shininess exponent

        // Transform
        TransformComponent transform;
    };

    // TODO possibly redundant
    struct RenderInformation {
        uint64_t photonsAccumulated = 0;
        uint64_t totalPhotons = 0;
        float gamma = 1.0f;
        uint32_t numBounces = 0;
        uint32_t frameID = 0;
        bool applyBetaWeight = false;
    };

    struct BVHNode {
        glm::vec3 bboxMin = glm::vec3(0.0f);
        glm::vec3 bboxMax = glm::vec3(0.0f);
        int leftChild = -1; // index into BVH nodes array (or -1 if leaf)
        int rightChild = -1; // index into BVH nodes array (or -1 if leaf)
        // For leaf nodes, you can store the quadric index.
        int quadricIndex = -1;
        bool isLeaf = false;
    };


    struct GPUData {
        // GS
        GaussianInputAssembly *gaussianInputAssembly = nullptr;
        size_t numGaussians = 0;

        // Quadric
        QuadricInputAssembly *quadricInputAssembly = nullptr;
        size_t numQuadrics = 0;
        size_t numEntities = 0;
        // QUadric BVH
        BVHNode *bvhNodes = nullptr;
        size_t numBVHNodes = 0;

        glm::vec3 *gradients = nullptr;
        //glm::mat3* quadricGradients = nullptr;
        glm::mat3 *photonIDGradient = nullptr;
        glm::vec2 *gradientPixelCoordinates = nullptr;
        glm::vec3 *gaussianGradients = nullptr;
        float *gradientImageU = nullptr;
        float *gradientImageV = nullptr;
        float *gradientImagePerObject = nullptr;

        float *imageMemory = nullptr;
        float *imageMemoryCounter = nullptr;
        float *imageMemoryPersistent = nullptr;

        PinholeCamera *pinholeCamera = nullptr;
        TransformComponent *cameraTransform = nullptr;
        RenderInformation *renderInformation = nullptr;
    };

    // Stored per photon
    struct GPUDataOutput {
        // Direct lighting parameters
        bool hitCamera = false;
        float emissionDirectionLength = 0.0f; // etmin

        size_t gaussianID = UINT64_MAX;

        glm::vec3 emissionOrigin = glm::vec3(0.0f); // eo
        glm::vec3 emissionDirection = glm::vec3(0.0f); // ed
        glm::vec3 apertureHitPoint = glm::vec3(0.0f); // a
        glm::vec3 cameraHitPointLocal = glm::vec3(0.0f); // p
        glm::vec3 directLightingDir = glm::vec3(0.0f); // p

        struct QuadraticInfo {
            float A = 0.0f;
            float B = 0.0f;
            float C = 0.0f;
            float discriminant = 0.0f;
            int rootIndex = -1; //-1 is no root, 1 is negative root, 2 is positive root
            float geodesic = -1.0f;
            float betaContribution = 0.0f;

            float rho = 0.0f;
            float theta = 0.0f;
            float a_theta = 0.0f;

            glm::vec3 localRayOrigin = glm::vec3(0.0f);
            glm::vec3 localRayDirection = glm::vec3(0.0f);
            glm::vec2 hitLocal = glm::vec2(0.0f);
        };

        struct Bounce {
            //Properties:
            size_t quadricID = UINT64_MAX;
            QuadraticInfo quadInfo;
            glm::vec3 hitPointWorld = glm::vec3(0.0f);
            glm::vec3 hitNormalWorld = glm::vec3(0.0f);
            glm::vec3 outGoingDirection = glm::vec3(0.0f);
            glm::vec3 outGoingOrigin = glm::vec3(0.0f);
            bool hitCamera = false;
            glm::vec3 apertureDirection = glm::vec3(0.0f); // ed
            glm::vec3 apertureHitPoint = glm::vec3(0.0f); // a
            glm::vec3 cameraHitPointLocal = glm::vec3(0.0f); // p
            float cameraDirectionLength = 0.0f; // etmin

            glm::vec2 pixelCoordinate = glm::vec2(0.0f);
            float emissionDirectionLength = 0.0f; // etmin
        };

        // 1 bounce
        Bounce bounce[2];
    };


    struct PCG32 {
        uint64_t state{};
        uint64_t inc{};

        // Initialize the RNG with a seed and sequence
        void init(uint64_t seed, uint64_t sequence = 1) {
            state = 0;
            inc = (sequence << 1u) | 1u; // Increment must be odd
            nextUInt(); // Advance state
            state += seed;
            nextUInt(); // Advance state again
        }

        // Generate the next uint32_t random number
        uint32_t nextUInt() {
            uint64_t old_state = state;
            state = old_state * 6364136223846793005ULL + inc;
            auto xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
            uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

        // Generate a random float in [0, 1)
        float nextFloat() {
            return static_cast<float>(nextUInt()) / static_cast<float>(UINT32_MAX);
        }
    };
}
#endif //DEFINITIONS_H
