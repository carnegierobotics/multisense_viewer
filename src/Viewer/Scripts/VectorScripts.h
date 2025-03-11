//
// Created by magnus on 3/10/25.
//

#ifndef VECTORSCRIPTS_H
#define VECTORSCRIPTS_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>

#include <Viewer/Rendering/Components/GaussianComponent.h>

#include "Viewer/Scenes/ScriptableEntity.h"


namespace VkRender {

class VectorScripts : ScriptableEntity {
    TransformComponent*transform = nullptr;
public:
    void onUpdate(Timestep ts) override {
        Log::Logger::getInstance()->info("OnUpdate");
        transform->setPosition(glm::vec3(3.0f, 0.0f, 0.0f));

        /*
        auto* scene = m_entity.getScene();
                Entity emissiveGaussianEntity;

        auto gaussianView = scene->getRegistry().view<GaussianComponent2DGS>();
        for (auto e: gaussianView) {
            // Wrap the entity to use our helper functions.
            Entity gaussianEntity(e, scene);
            auto &gaussianComp = gaussianEntity.getComponent<GaussianComponent2DGS>();
            // For this example, we use the first position and normal from the component.
            size_t numGaussians = gaussianComp.size();
            for (size_t i = 0; i < numGaussians; i++) {
                glm::vec3 position = gaussianComp.positions[i];
                glm::vec3 normal = gaussianComp.normals[i];
                glm::vec2 scale = gaussianComp.scales[i];
                float emission = gaussianComp.emissions[i];
                float specular = gaussianComp.specular[i];
                float diffuse = gaussianComp.diffuse[i];
                // Create a unique name for the mesh entity associated with this gaussian.
                std::string entityName = "GaussianEntity_" + std::to_string(static_cast<uint32_t>(i));
                // Get or create the entity with the given name.
                Entity meshEntity = scene->getOrCreateEntityByName(entityName);
                // Only add the MeshComponent if it doesn't already exist.
                if (!meshEntity.hasComponent<MeshComponent>()) {
                    meshEntity.addComponent<MeshComponent>(
                        OBJ_FILE,
                        "../Resources/models-repository/disk.obj"
                    );
                }

                if (emission > 0.0f) {
                    auto &light = meshEntity.getOrAddComponent<LightSourceComponent>();
                    light.position = position;
                    light.normal = normal;
                    emissiveGaussianEntity = gaussianEntity;
                } else {
                    if (meshEntity.hasComponent<LightSourceComponent>())
                        meshEntity.removeComponent<LightSourceComponent>();
                }


                // Only add the MaterialComponent if it doesn't already exist.
                if (!meshEntity.hasComponent<MaterialComponent>()) {
                    meshEntity.addComponent<MaterialComponent>();
                }
                // Ensure that a TransformComponent exists.
                if (!meshEntity.hasComponent<TransformComponent>()) {
                    meshEntity.addComponent<TransformComponent>();
                }
                if (!meshEntity.hasComponent<TemporaryComponent>()) {
                    meshEntity.addComponent<TemporaryComponent>();
                }

                auto &transform = meshEntity.getComponent<TransformComponent>();
                // Update the transform's position.
                transform.setPosition(position);
                transform.setScale({scale.x, scale.y, 1.0f});
                // Compute the quaternion rotation so that the local up vector (0,1,0)
                // aligns with the Gaussian normal.
                glm::vec3 localUp(0.0f, 0.0f, 1.0f);
                glm::quat rotation = glm::normalize(glm::rotation(localUp, glm::normalize(normal)));
                transform.setRotationQuaternion(rotation);

                auto &material = meshEntity.getComponent<MaterialComponent>();
                material.emission = emission;
                material.specular = specular;
                material.diffuse = diffuse;
                material.albedo = gaussianComp.colors[i];
            }
        }

        Entity quadricEntity;
        auto quadricView = scene->getRegistry().view<MeshComponent>();
        for (auto e: quadricView) {
            // Wrap the entity to use our helper functions.
            Entity entity(e, scene);

            auto &meshComponent = entity.getComponent<MeshComponent>();
            if (meshComponent.meshDataType() == QUADRIC && entity.getName() == "Quadric2") {
                quadricEntity = entity;
            }
        }

            bool updateOnEachFrame = false;
        if (!updateOnEachFrame) {
            return;
        }
        glm::vec3 g_hit(0.0f);
        glm::vec3 e_d(0.0f);
        glm::vec3 g_hit_gt(0.0f);
        glm::vec3 e_d_gt(0.0f);
        auto rayView = scene->getRegistry().view<MeshComponent>();
        for (auto e: rayView) {
            Entity entity(e, scene);
            if (entity.getName() == "e_d") {
                auto &emissiveRayTransform = entity.getComponent<TransformComponent>();
                auto &emissiveRayMesh = entity.getComponent<MeshComponent>();
                auto emissiveRayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(
                    emissiveRayMesh.meshParameters);

                auto &quadricTransform = quadricEntity.getComponent<TransformComponent>();
                auto quadricMesh = quadricEntity.getComponent<MeshComponent>();
                auto quadricParams = std::dynamic_pointer_cast<QuadricMeshParameters>(quadricMesh.meshParameters);

                float a = quadricParams->a;
                float b = quadricParams->b;
                float c = quadricParams->c;
                float alpha_x = tanh(quadricParams->t_x);
                float alpha_y = tanh(quadricParams->t_y);
                glm::vec3 g_c = quadricTransform.getPosition();
                glm::vec3 e_o = emissiveGaussianEntity.getComponent<TransformComponent>().getPosition();
                //e_d = glm::normalize(glm::vec3(-0.1f, 0.0f, -1.0f));
                e_d = emissiveRayParams->direction;
                e_o = emissiveRayParams->origin;
                //emissiveRayParams->origin = e_o;
                // = e_d;

                // Compute the world hit point.
                g_hit = computeWorldHitPoint(e_o, e_d, g_c, a, b, c, alpha_x, alpha_y);
                float length = glm::length(g_hit - e_o);
                emissiveRayParams->magnitude = length;
                emissiveRayMesh.updateMeshData = updateOnEachFrame;
            }
            if (entity.getName() == "e_d_gt") {
                auto &emissiveRayTransform = entity.getComponent<TransformComponent>();
                auto &emissiveRayMesh = entity.getComponent<MeshComponent>();
                auto emissiveRayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(
                    emissiveRayMesh.meshParameters);

                auto &quadricTransform = quadricEntity.getComponent<TransformComponent>();
                auto quadricMesh = quadricEntity.getComponent<MeshComponent>();
                auto quadricParams = std::dynamic_pointer_cast<QuadricMeshParameters>(quadricMesh.meshParameters);

                float a = quadricParams->a;
                float b = quadricParams->b;
                float c = quadricParams->c;
                float alpha_x = tanh(quadricParams->t_x);
                float alpha_y = tanh(quadricParams->t_y);
                glm::vec3 g_c = quadricTransform.getPosition();
                glm::vec3 e_o = emissiveGaussianEntity.getComponent<TransformComponent>().getPosition();
                e_d_gt = glm::normalize(g_c - e_o);
                // = e_d_gt;

                // Compute the world hit point.
                g_hit_gt = computeWorldHitPoint(e_o, e_d_gt, g_c, a, b, c, alpha_x, alpha_y);
                float length = glm::length(g_hit_gt - e_o);
                emissiveRayParams->direction = e_d_gt;
                emissiveRayParams->origin = e_o;
                emissiveRayParams->magnitude = length;
                emissiveRayMesh.updateMeshData = updateOnEachFrame;
            }
        }

        // Second pass: process the aperture ray ("a_d") using the computed g_hit.
        std::string cameraName = "Camera1";
        for (auto e: rayView) {
            Entity entity(e, scene);

            if (entity.getName() == "a_d") {
                auto &apertureRayTransform = entity.getComponent<TransformComponent>();
                auto &apertureRayMesh = entity.getComponent<MeshComponent>();
                auto apertureRayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(
                    apertureRayMesh.meshParameters);

                apertureRayParams->origin = g_hit;

                glm::vec3 a_c(0.0f);
                TransformComponent cameraTransform;
                auto apertureView = scene->getRegistry().view<CameraComponent>();
                for (auto ent: apertureView) {
                    auto entt = Entity(ent, scene);
                    if (entt.getName() != cameraName)
                        continue;
                    a_c = entt.getComponent<TransformComponent>().getPosition();
                    cameraTransform = entt.getComponent<TransformComponent>();
                    break;
                }

                glm::vec3 a_d = glm::normalize(a_c - g_hit);
                apertureRayParams->direction = a_d;

                auto camera2World = cameraTransform.getTransform();
                glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
                glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
                glm::vec3 f = cameraPlanePointWorld;
                glm::vec3 f_n = cameraNormal;

                float a_tmin = glm::dot(f - g_hit, f_n) / glm::dot(a_d, f_n);
                apertureRayParams->magnitude = a_tmin;
                apertureRayMesh.updateMeshData = updateOnEachFrame;
            }

            if (entity.getName() == "a_d_gt") {
                auto &apertureRayGTTransform = entity.getComponent<TransformComponent>();
                auto &apertureRayGTMesh = entity.getComponent<MeshComponent>();
                auto apertureRayGTParams = std::dynamic_pointer_cast<CylinderMeshParameters>(
                    apertureRayGTMesh.meshParameters);

                apertureRayGTParams->origin = g_hit_gt;

                glm::vec3 a_c(0.0f);
                TransformComponent cameraTransform;
                auto apertureView = scene->getRegistry().view<CameraComponent>();
                for (auto ent: apertureView) {
                    auto entt = Entity(ent, scene);
                    if (entt.getName() != cameraName)
                        continue;
                    a_c = entt.getComponent<TransformComponent>().getPosition();
                    cameraTransform = entt.getComponent<TransformComponent>();
                    break;
                }

                glm::vec3 a_d = glm::normalize(a_c - g_hit_gt);
                apertureRayGTParams->direction = a_d;

                auto camera2World = cameraTransform.getTransform();
                glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
                glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
                glm::vec3 f = cameraPlanePointWorld;
                glm::vec3 f_n = cameraNormal;

                float a_tmin = glm::dot(f - g_hit_gt, f_n) / glm::dot(a_d, f_n);
                apertureRayGTParams->magnitude = a_tmin;
                apertureRayGTMesh.updateMeshData = updateOnEachFrame;
            }

            if (entity.getName() == "g_hit") {
                auto &quadricTransform = quadricEntity.getComponent<TransformComponent>();
                auto quadricMesh = quadricEntity.getComponent<MeshComponent>();
                auto quadricParams = std::dynamic_pointer_cast<QuadricMeshParameters>(quadricMesh.meshParameters);

                auto &gHitRayTransform = entity.getComponent<TransformComponent>();
                auto &gHitRayMesh = entity.getComponent<MeshComponent>();
                auto gHitRayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(gHitRayMesh.meshParameters);
                float alpha_x = tanh(quadricParams->t_x);
                float alpha_y = tanh(quadricParams->t_y);
                // 7) Compute local normal from gradient: ∇f(x,y,z) = (2*c*alphaX*x/a^2, 2*c*alphaY*y/b^2, -1)
                glm::vec3 gradLocal(
                    2.0f * quadricParams->c * alpha_x * g_hit.x / (quadricParams->a * quadricParams->a),
                    2.0f * quadricParams->c * alpha_y * g_hit.y / (quadricParams->b * quadricParams->b),
                    -1.0f
                );
                // 8) Transform the local normal to world space
                //    If your transform is just rotation+translation (orthonormal),
                //    you can multiply by the rotation part. For a general affine transform,
                //    the correct approach is n_world = normalize( inverseTranspose(M) * n_local ).
                glm::mat3 mat = glm::mat3(quadricTransform.getTransform());
                // Now safe to compute:
                glm::mat3 invT = glm::inverseTranspose(mat);
                glm::vec3 normalW = glm::normalize(invT * gradLocal);

                if (glm::dot(normalW, e_d) > 0.0f)
                    normalW = -normalW;


                gHitRayParams->origin = g_hit;
                gHitRayParams->direction = normalW;
                gHitRayParams->magnitude = 1.0f;
                gHitRayMesh.updateMeshData = updateOnEachFrame;
            }
            if (entity.getName() == "p") {
                auto &quadricTransform = quadricEntity.getComponent<TransformComponent>();
                auto quadricMesh = quadricEntity.getComponent<MeshComponent>();
                auto quadricParams = std::dynamic_pointer_cast<QuadricMeshParameters>(quadricMesh.meshParameters);

                auto &pHitTransform = entity.getComponent<TransformComponent>();
                auto &pHitMesh = entity.getComponent<MeshComponent>();
                auto pHitParams = std::dynamic_pointer_cast<CylinderMeshParameters>(pHitMesh.meshParameters);

                glm::vec3 a_c(0.0f);
                TransformComponent cameraTransform;
                auto apertureView = scene->getRegistry().view<CameraComponent>();
                for (auto ent: apertureView) {
                    auto entt = Entity(ent, scene);
                    a_c = entt.getComponent<TransformComponent>().getPosition();
                    cameraTransform = entt.getComponent<TransformComponent>();
                    break;
                }

                glm::vec3 a_d = glm::normalize(a_c - g_hit);
                auto camera2World = cameraTransform.getTransform();
                glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
                glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
                glm::vec3 f = cameraPlanePointWorld;
                glm::vec3 f_n = cameraNormal;

                float a_tmin = glm::dot(f - g_hit, f_n) / glm::dot(a_d, f_n);

                glm::vec3 p = g_hit + a_tmin * a_d;

                pHitParams->origin = p;
                pHitParams->direction = f_n;
                //pHitParams->magnitude = 0.1f;
                pHitMesh.updateMeshData = updateOnEachFrame;
            }
        }
        */
    }

    void onDestroy() override {
        Log::Logger::getInstance()->info("OnDestroy");

    }

    void onCreate() override {
        transform = &getComponent<TransformComponent>();
        Log::Logger::getInstance()->info("OnCreate");

    }


};

}

#endif //VECTORSCRIPTS_H
