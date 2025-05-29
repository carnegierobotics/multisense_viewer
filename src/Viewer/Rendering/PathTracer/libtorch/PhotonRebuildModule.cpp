//
// Created by magnus on 1/24/25.
//

#include "Viewer/Rendering/PathTracer/libtorch/PhotonRebuildModule.h"

#include <utility>

#include "Viewer/Rendering/Components/LightSourceComponent.h"
#include "Viewer/Scenes/Entity.h"

namespace VkRender::PathTracer {
    PhotonRebuildModule::PhotonRebuildModule(std::weak_ptr<Scene> scene) {
        // Optionally register parameters or buffers if needed
        //uploadTensorFromScene(std::move(scene));
    }

    PhotonRebuildModule::~PhotonRebuildModule() {
        freeData();
    }

    torch::Tensor
    PhotonRebuildModule::forward() {
        // 1) Call PhotonTracer::update(...) or any function you want to do the actual path tracing
        //    In your case:  rt_->update(...);


        return torch::Tensor();
    }

    float *PhotonRebuildModule::getRenderedImage() {
        if (m_outputTensor.defined()) {
            return m_outputTensor.data_ptr<float>(); // Get a float pointer to the tensor data
        }
        return nullptr;
    }

    void PhotonRebuildModule::freeData() {
    }

    void PhotonRebuildModule::uploadPathTracerFromTensor() {
    }

    void PhotonRebuildModule::uploadSceneFromTensor(std::shared_ptr<Scene> scene) {
        /*
        // Get views of all the 2DGS Gaussian components in the scene.
        auto gaussianView = scene->getRegistry().view<LightSourceComponent>();

        // Get CPU copies of our tensors (if they aren’t already on CPU)
        auto positionsTensor = m_tensorData.positions.cpu();
        auto scalesTensor = m_tensorData.scales.cpu();
        auto normalsTensor = m_tensorData.normals.cpu();
        auto emissionsTensor = m_tensorData.emissions.cpu();
        auto colorsTensor = m_tensorData.colors.cpu();
        auto specularTensor = m_tensorData.specular.cpu();
        auto diffuseTensor = m_tensorData.diffuse.cpu();

        // Assume that the first dimension of each tensor is the number of gaussians.
        int numGaussians = positionsTensor.size(0);

        // Pointers to the raw data.
        // (These assume that the tensors are contiguous and of type float.)
        float* posPtr = positionsTensor.data_ptr<float>(); // shape: [numGaussians, 3]
        float* scalePtr = scalesTensor.data_ptr<float>(); // shape: [numGaussians, 2]
        float* normPtr = normalsTensor.data_ptr<float>(); // shape: [numGaussians, 3]
        float* emissPtr = emissionsTensor.data_ptr<float>(); // shape: [numGaussians]
        float* colorPtr = colorsTensor.data_ptr<float>(); // shape: [numGaussians]
        float* specPtr = specularTensor.data_ptr<float>(); // shape: [numGaussians]
        float* diffPtr = diffuseTensor.data_ptr<float>(); // shape: [numGaussians]

        // For each LightSourceComponent in our scene, update its vectors with the tensor data.
        // (Often in an ECS there is only one global component of a given type,
        //  but if there are multiple, they will all be updated identically.)
        for (auto entityID : gaussianView) {
            auto entity = Entity(entityID, scene.get());
            auto& comp = entity.getComponent<LightSourceComponent>();

            // Resize the vectors to hold data for all gaussians.
            comp.positions.resize(numGaussians);
            comp.scales.resize(numGaussians);
            comp.normals.resize(numGaussians);
            comp.emissions.resize(numGaussians);
            comp.colors.resize(numGaussians);
            comp.specular.resize(numGaussians);
            comp.diffuse.resize(numGaussians);
            // Note: If opacities or phongExponents should also be updated,
            // add them here and ensure the corresponding tensors exist.

            // Copy the data from the tensors to the component's vectors.
            for (int i = 0; i < numGaussians; i++) {
                // Update positions (each is 3 floats)
                comp.positions[i] = glm::vec3(
                    posPtr[i * 3 + 0],
                    posPtr[i * 3 + 1],
                    posPtr[i * 3 + 2]
                );
                // Update scales (each is 2 floats)
                comp.scales[i] = glm::vec2(
                    scalePtr[i * 2 + 0],
                    scalePtr[i * 2 + 1]
                );
                // Update normals (each is 3 floats)
                comp.normals[i] = glm::vec3(
                    normPtr[i * 3 + 0],
                    normPtr[i * 3 + 1],
                    normPtr[i * 3 + 2]
                );
                // Update colors (each is 4 floats)
                comp.colors[i] = glm::vec4(
                    colorPtr[i * 4 + 0],
                    colorPtr[i * 4 + 1],
                    colorPtr[i * 4 + 2],
                    colorPtr[i * 4 + 3]
                );

                // Update the other properties (each assumed to be a single float per gaussian)
                comp.emissions[i] = emissPtr[i];
                comp.specular[i] = specPtr[i];
                comp.diffuse[i] = diffPtr[i];
            }
        }

        float* quadricsPtr = m_tensorData.quadrics.cpu().data_ptr<float>(); // shape: [numGaussians]
        float* quadricsPosPtr = m_tensorData.quadricPositions.cpu().data_ptr<float>(); // shape: [numGaussians]
        float* quadricsRotPtr = m_tensorData.quadricRotations.cpu().data_ptr<float>(); // shape: [numGaussians]


        // Update the optimization variables
        auto viewQuadric = scene->getRegistry().view<MeshComponent, MaterialComponent>();
        for (int i = 0; auto e : viewQuadric) {
            auto& component = Entity(e, scene.get()).getComponent<MeshComponent>();
            auto& material = Entity(e, scene.get()).getComponent<MaterialComponent>();
            auto& transform = Entity(e, scene.get()).getComponent<TransformComponent>();
            if (component.meshDataType() == QUADRIC) {
                auto parameters = std::dynamic_pointer_cast<QuadricMeshParameters>(component.meshParameters);
                if (!parameters) {
                    continue;
                }
                glm::vec3 translation = {quadricsPosPtr[i * 3 + 0], quadricsPosPtr[i * 3 + 1], quadricsPosPtr[i * 3 + 2]};
                transform.setPosition(translation);
                glm::quat quat = glm::quat(quadricsRotPtr[i * 4 + 0], quadricsRotPtr[i * 4 + 1], quadricsRotPtr[i * 4 + 2], quadricsRotPtr[i * 4 + 3]);
                transform.setRotationQuaternion(quat);

                parameters->a = quadricsPtr[i * 12 + 0];
                parameters->b = quadricsPtr[i * 12 + 1];
                parameters->c = quadricsPtr[i * 12 + 2];
                parameters->t_x = quadricsPtr[i * 12 + 3];
                parameters->t_y = quadricsPtr[i * 12 + 4];
                parameters->b_beta = quadricsPtr[i * 12 + 5];
                parameters->threshold = quadricsPtr[i * 12 + 6];
                parameters->kernelScale = quadricsPtr[i * 12 + 7];
                parameters->min.x = quadricsPtr[i * 12 + 8];
                parameters->max.x = quadricsPtr[i * 12 + 9];
                parameters->min.y = quadricsPtr[i * 12 + 10];
                parameters->max.y = quadricsPtr[i * 12 + 11];

                material.emission = 0.0f;
                material.diffuse = 1.0f;
                material.specular = 0.0f;
                material.phongExponent = 32.0f;
                material.albedo = glm::vec4(0.8f);


                i++;
            }
        }
        */
    }


    void PhotonRebuildModule::uploadTensorFromScene(std::weak_ptr<Scene> scene) {
        /*
        freeData();
        auto scenePtr = scene.lock();
        std::vector<GaussianInputAssembly> gaussianInputAssembly;
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        auto& registry = scenePtr->getRegistry();
        // Find all entities with GaussianComponent
        auto view = registry.view<LightSourceComponent>();
        for (auto e : view) {
            auto& component = Entity(e, scenePtr.get()).getComponent<LightSourceComponent>();
            for (size_t i = 0; i < component.size(); ++i) {
                GaussianInputAssembly point{};
                point.position = component.positions[i];
                point.scale = component.scales[i];
                point.normal = component.normals[i];

                point.emission = component.emissions[i];
                point.color = component.colors[i];
                point.diffuse = component.diffuse[i];
                point.specular = component.specular[i];
                point.phongExponent = component.phongExponents[i];
                gaussianInputAssembly.push_back(point);
            }
            auto& transform = Entity(e, scenePtr.get()).getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);
        }

        m_data.gaussianInputAssembly = static_cast<GaussianInputAssembly*>(malloc(
            sizeof(LightSourceComponent) * gaussianInputAssembly.size()));
        memcpy(m_data.gaussianInputAssembly, gaussianInputAssembly.data(),
               sizeof(LightSourceComponent) * gaussianInputAssembly.size());

        m_data.numGaussians = gaussianInputAssembly.size(); // Number of entities for rendering


        // Now we have gaussianInputAssembly filled. Suppose we want to create separate PyTorch Tensors:
        auto device = torch::kCPU; // or torch::kCPU, depends on your use-case

        // 1) Convert positions to a tensor of shape [N, 3]
        std::vector<float> hostPositions;
        hostPositions.reserve(gaussianInputAssembly.size() * 3);

        for (auto& item : gaussianInputAssembly) {
            hostPositions.push_back(item.position.x);
            hostPositions.push_back(item.position.y);
            hostPositions.push_back(item.position.z);
        }

        // from_blob does not copy by default. Once we go out of scope, hostPositions might be freed.
        // Usually, we wrap it in a clone() call to own the data inside a Torch tensor:
        m_tensorData.positions = torch::from_blob(
            hostPositions.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(true);

        // 2) Convert scales to a tensor of shape [N, 1]
        std::vector<glm::vec2> hostScales;
        hostScales.reserve(gaussianInputAssembly.size());
        for (auto& item : gaussianInputAssembly) {
            hostScales.push_back(item.scale);
        }

        m_tensorData.scales = torch::from_blob(
            hostScales.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 2},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        // ... do the same for normals, emissions, colors, etc. ...

        // Example for normals:
        std::vector<float> hostNormals;
        hostNormals.reserve(gaussianInputAssembly.size() * 3);
        for (auto& item : gaussianInputAssembly) {
            hostNormals.push_back(item.normal.x);
            hostNormals.push_back(item.normal.y);
            hostNormals.push_back(item.normal.z);
        }

        m_tensorData.normals = torch::from_blob(
            hostNormals.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        // Example for normals:
        std::vector<float> emissions;
        std::vector<glm::vec4> colors;
        std::vector<float> specular;
        std::vector<float> diffuse;
        emissions.reserve(gaussianInputAssembly.size());
        for (auto& item : gaussianInputAssembly) {
            emissions.push_back(item.emission);
            colors.push_back(item.color);
            specular.push_back(item.specular);
            diffuse.push_back(item.diffuse);
        }

        m_tensorData.emissions = torch::from_blob(
            emissions.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        m_tensorData.colors = torch::from_blob(
            colors.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        m_tensorData.specular = torch::from_blob(
            specular.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        m_tensorData.diffuse = torch::from_blob(
            diffuse.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);


        register_parameter("positions", m_tensorData.positions);
        register_parameter("scales", m_tensorData.scales);
        register_parameter("normals", m_tensorData.normals);

        register_parameter("emissions", m_tensorData.emissions);
        register_parameter("colors", m_tensorData.colors);
        register_parameter("specular", m_tensorData.scales);
        register_parameter("diffuse", m_tensorData.diffuse);

        Log::Logger::getInstance()->info("Registrered and Uploaded {} Gaussians from scene to Tensors",
                                         m_data.numGaussians);

        std::vector<QuadricInputAssembly> quadricInputAssembly;
        std::vector<TransformComponent> quadricTransformMatrices; // Transformation matrices for entities
        // Find all entities with GaussianComponent
        auto viewQuadric = scenePtr->getRegistry().view<MeshComponent, MaterialComponent>();
        for (auto e : viewQuadric) {
            auto& component = Entity(e, scenePtr.get()).getComponent<MeshComponent>();
            auto& material = Entity(e, scenePtr.get()).getComponent<MaterialComponent>();
            if (component.meshDataType() == QUADRIC) {
                auto parameters = std::dynamic_pointer_cast<QuadricMeshParameters>(component.meshParameters);
                if (!parameters) {
                    continue;
                }
                QuadricInputAssembly point{};
                point.a = parameters->a;
                point.b = parameters->b;
                point.c = parameters->c;
                point.t_x = parameters->t_x;
                point.t_y = parameters->t_y;
                point.min = parameters->min;
                point.max = parameters->max;

                point.b_beta = parameters->b_beta;
                point.threshold = parameters->threshold;
                point.kernelScale = parameters->kernelScale;

                point.emission = material.emission;
                point.color = material.albedo;
                point.diffuse = material.diffuse;
                point.specular = material.specular;
                point.phongExponent = material.phongExponent;
                quadricInputAssembly.push_back(point);
                auto& transform = Entity(e, scenePtr.get()).getComponent<TransformComponent>();
                quadricTransformMatrices.emplace_back(transform);
            }
        }
        // 1) Convert to a tensor of shape [N, 12]
        std::vector<float> hostQuadrics;
        hostQuadrics.reserve(quadricInputAssembly.size() * 12);

        for (auto& item : quadricInputAssembly) {
            hostQuadrics.push_back(item.a);
            hostQuadrics.push_back(item.b);
            hostQuadrics.push_back(item.c);
            hostQuadrics.push_back(item.t_x);
            hostQuadrics.push_back(item.t_y);
            hostQuadrics.push_back(item.b_beta);
            hostQuadrics.push_back(item.threshold);
            hostQuadrics.push_back(item.kernelScale);
            hostQuadrics.push_back(item.min.x);
            hostQuadrics.push_back(item.max.x);
            hostQuadrics.push_back(item.min.y);
            hostQuadrics.push_back(item.max.y);
        }

        // from_blob does not copy by default. Once we go out of scope, hostQuadrics might be freed.
        // Usually, we wrap it in a clone() call to own the data inside a Torch tensor:
        m_tensorData.quadrics = torch::from_blob(
            hostQuadrics.data(),
            {static_cast<long>(quadricInputAssembly.size()), 12},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        // 1) Convert to a tensor of shape [N, 12]
        std::vector<float> hostQuadricPositions;
        hostQuadricPositions.reserve(quadricTransformMatrices.size() * 3);

        for (auto& item : quadricTransformMatrices) {
            hostQuadricPositions.push_back(item.getPosition().x);
            hostQuadricPositions.push_back(item.getPosition().y);
            hostQuadricPositions.push_back(item.getPosition().z);
        }

        // from_blob does not copy by default. Once we go out of scope, hostQuadrics might be freed.
        // Usually, we wrap it in a clone() call to own the data inside a Torch tensor:
        m_tensorData.quadricPositions = torch::from_blob(
            hostQuadricPositions.data(),
            {static_cast<long>(quadricTransformMatrices.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(true);

        // 1) Convert to a tensor of shape [N, 8]
        std::vector<float> hostQuadricRotations;
        hostQuadricRotations.reserve(quadricTransformMatrices.size() * 4);

        for (auto& item : quadricTransformMatrices) {
            hostQuadricRotations.push_back(item.getRotationQuaternion().w);
            hostQuadricRotations.push_back(item.getRotationQuaternion().x);
            hostQuadricRotations.push_back(item.getRotationQuaternion().y);
            hostQuadricRotations.push_back(item.getRotationQuaternion().z);
        }

        m_tensorData.quadricRotations = torch::from_blob(
            hostQuadricRotations.data(),
            {static_cast<long>(quadricTransformMatrices.size()), 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        register_parameter("quadrics", m_tensorData.quadrics);
        register_parameter("quadricPositions", m_tensorData.quadricPositions);
        register_parameter("quadricRotations", m_tensorData.quadricRotations);

        Log::Logger::getInstance()->info("Registrered and Uploaded {} Quadrics from scene to Tensors",
                                         m_data.numQuadrics);
        */
    }
}
