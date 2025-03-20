//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include <utility>

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/MeshManager.h"

#ifdef SYCL_ENABLED
#include "Viewer/Tools/SYCLDeviceSelector.h"
#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"
#endif

#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"

namespace VkRender::PathTracer {
#ifdef SYCL_ENABLED



    class PhotonTracer {
    public:
        struct RenderSettings {
            PinholeCamera camera{};
            TransformComponent cameraTransform{};

            KernelType kernelType = KERNEL_PATH_TRACER_2DGS;

            // Render settings
            float gammaCorrection = 2.2f;
            bool applyBetaContribution = true;
        };

        struct PipelineSettings {
            uint32_t width = 0;
            uint32_t height = 0;

            uint64_t photonCount = 1e4;
            uint32_t numBounces = 1;
            uint32_t numFrames = 1;

            std::shared_ptr<SYCLDeviceSelector> syclDevice;

            [[nodiscard]] sycl::queue& device() const {
                return syclDevice->getQueue();
            }
            PipelineSettings(std::shared_ptr<SYCLDeviceSelector> q, uint32_t width, uint32_t height, uint64_t photonCount = 1e4, uint32_t numBounces = 1) :
                syclDevice(std::move(q)), width(width), height(height), photonCount(photonCount), numBounces(numBounces) {
            }
        };

        struct BackwardInfo {
            glm::vec3* gradients = nullptr;
            glm::mat3* photonIDGradient = nullptr;
            glm::vec2* gradientPixelCoordinates = nullptr;
            std::vector<glm::mat3> quadricGradients;
            float* gradientImage = nullptr;
            float* gradientImagePerObject = nullptr;
        };

        struct BVHLeaf {
            glm::vec3 bboxMin;
            glm::vec3 bboxMax;
            size_t quadricIndex;
        };


        struct AABB {
            glm::vec3 min;
            glm::vec3 max;
        };


        PhotonTracer(Application* context, const PipelineSettings& pipelineSettings, std::shared_ptr<Scene> scene);

        void setExecutionDevice();

        void update(RenderSettings& editorImageUI);
        BackwardInfo backward(RenderSettings& settings);

        void resetImage();
        float* getImage() { return m_imageMemory; }

        ~PhotonTracer();


        const PipelineSettings& getPipelineSettings() {
            return m_pipelineSettings;
        }
        RenderInformation* getRenderInfo() {
            return m_renderInformation.get();
        }
        void uploadGaussiansFromTensors(GPUDataTensors& data);


        BackwardInfo m_backwardInfo; // TODO make private once we figure out the strucutre of this.

    private:
        PipelineSettings m_pipelineSettings;

        Application* m_context;
        std::unique_ptr<RenderInformation> m_renderInformation;

        float* m_imageMemory = nullptr;

        GPUData m_gpu;
        PCG32* m_pcg32 = nullptr;
        GPUDataOutput* m_gpuDataOutput = nullptr;


        void saveAsPPM(const std::filesystem::path& filename) const;
        void saveAsPFM(const std::filesystem::path& filename) const;
        void freeResources();
        void prepareImageAndInfoBuffers();
        void uploadGaussianData(std::shared_ptr<Scene>& scene);

        void uploadQuadricEntities(std::shared_ptr<Scene> &scene);

        void uploadVertexData(std::shared_ptr<Scene>& scene);
        std::vector<BVHLeaf> buildBVHLeaves(const std::vector<QuadricInputAssembly>& quadrics);
        std::vector<BVHNode> buildBVH(const std::vector<BVHLeaf>& inputLeaves);
        float computeLocalZ(float x, float y, const QuadricInputAssembly& quadric);
        AABB computeLocalAABB(const QuadricInputAssembly& quadric);
    };

#else

    class PhotonTracer {
    public:

        struct RenderSettings {
            PinholeCamera camera{};
            TransformComponent cameraTransform{};

            KernelType kernelType = KERNEL_PATH_TRACER_2DGS;

            // Render settings
            float gammaCorrection = 2.2f;
        };


        PhotonTracer(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) {}
        void uploadGaussianData(std::shared_ptr<Scene>& scene) {}
        void uploadVertexData(std::shared_ptr<Scene>& scene) {}
        void update(const EditorPathTracerLayerUI& editorImageUI, std::shared_ptr<Scene> scene) {}
        float* getImage() {return nullptr;}
        ~PhotonTracer() {}
        void upload(std::weak_ptr<Scene> scene, const RenderSettings& settings) {}
        void setActiveCamera(const TransformComponent &transformComponent, float w, float h){}
        void setActiveCamera(const std::shared_ptr<PinholeCamera>& camera, const TransformComponent *cameraTransform){}
    };
#endif


    struct IterationInfo {
        PathTracer::PhotonTracer::RenderSettings renderSettings;
        uint32_t iteration = 0;
        bool denoise = false;
        std::string cameraName;
    };
}


#endif //MULTISENSE_VIEWER_RAYTRACER_H
