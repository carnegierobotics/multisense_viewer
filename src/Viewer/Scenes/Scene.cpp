//
// Created by magnus on 8/13/24.
//

#include "Viewer/Scenes/Entity.h"

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Scenes/Controller.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>

#include <Viewer/Rendering/Components/GaussianComponent.h>

#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Components/ImageComponent.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"
#include "Viewer/Rendering/Components/QuadricCollectionComponent.h"
#include "Viewer/Rendering/Components/ScriptableComponent.h"

namespace VkRender {


    Scene::Scene(VkRender::Application *context) {
        m_context = context;

        auto entity = createEntity("NewEntity");
        auto& mesh = entity.addComponent<MeshComponent>(QUADRIC);
        entity.addComponent<MaterialComponent>();
        entity.addComponent<ScriptableComponent>().bind<Controller>();

    }

    static glm::vec3 computeWorldHitPoint(const glm::vec3 &e_o, const glm::vec3 &e_d,
                                          const glm::vec3 &g_c,
                                          float a, float b, float c,
                                          float alpha_x, float alpha_y) {
        // Since R_w2g is the identity, the local coordinates are simply:
        //   local direction: e_d,l,gt = e_d
        //   local origin:    e_o,l,gt = e_o - g_c
        glm::vec3 e_d_l = e_d;
        glm::vec3 e_o_l = e_o - g_c;

        // Compute the intersection parameters.
        // Note: In your provided printout, A turns out to be zero because the x,y components of e_d are zero.
        float A = c * (alpha_x * (e_d_l.x * e_d_l.x) / (a * a) +
                       alpha_y * (e_d_l.y * e_d_l.y) / (b * b));

        float B = c * (2.0f * alpha_x * (e_o_l.x * e_d_l.x) / (a * a) +
                       2.0f * alpha_y * (e_o_l.y * e_d_l.y) / (b * b))
                  - e_d_l.z;

        float C = c * (alpha_x * (e_o_l.x * e_o_l.x) / (a * a) +
                       alpha_y * (e_o_l.y * e_o_l.y) / (b * b))
                  - e_o_l.z;

        // In the provided code, the quadratic term A is zero (or negligible)
        // so the intersection parameter is computed as:
        float disc = (B * B) - 4 * A * C;

        // Solve for the smallest positive t (g_tmin)
        float g_tmin = 0.0f;
        if (abs(A) <= 0.0f) {
            g_tmin = -C/B;
        } else {
            g_tmin = (-B + std::sqrt(disc)) / (2.0f * A);
        }

        // Compute the local hit point: g_hit,l,gt = e_d,l,gt * t + e_o,l,gt
        glm::vec3 g_hit_l = e_d_l * g_tmin + e_o_l;

        // World hit point is then given by (R_g2w * local_point + g_c).
        // Since R_g2w is the identity, we simply add g_c.
        glm::vec3 g_hit = g_hit_l + g_c;

        return g_hit;
    }

    void Scene::update() {

        auto scriptView = m_registry.view<ScriptableComponent>();
        for (auto e : scriptView) {
            auto entity = Entity(e, this);
            auto& script = entity.getComponent<ScriptableComponent>();

            if (!script.instance) {
                script.instance = script.instantiateScript();
                script.instance->m_entity = entity;
                script.instance->onCreate();
            }
            script.instance->onUpdate();
        }

        auto cameraView = m_registry.view<CameraComponent>();
        for (auto e: cameraView) {
            auto entity = Entity(e, this);
            auto cameraComponent = entity.getComponent<CameraComponent>();
            auto &transform = entity.getComponent<TransformComponent>();
            cameraComponent.camera->updateViewMatrix(transform.getTransform());
            cameraComponent.camera->updateProjectionMatrix();

            if (entity.hasComponent<MaterialComponent>()) {
                auto &material = entity.getComponent<MaterialComponent>();
                if (cameraComponent.isActiveCamera()) {
                    material.albedo = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
                } else {
                    material.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
                }
            }
        }
        Entity emissiveGaussianEntity;
        auto gaussianView = m_registry.view<GaussianComponent2DGS>();
        for (auto e: gaussianView) {
            // Wrap the entity to use our helper functions.
            Entity gaussianEntity(e, this);
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
                Entity meshEntity = getOrCreateEntityByName(entityName);
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
        auto quadricView = m_registry.view<MeshComponent>();
        for (auto e: quadricView) {
            // Wrap the entity to use our helper functions.
            Entity entity(e, this);

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
        auto rayView = m_registry.view<MeshComponent>();
        for (auto e: rayView) {
            Entity entity(e, this);
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
            Entity entity(e, this);

            if (entity.getName() == "a_d") {
                auto &apertureRayTransform = entity.getComponent<TransformComponent>();
                auto &apertureRayMesh = entity.getComponent<MeshComponent>();
                auto apertureRayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(
                    apertureRayMesh.meshParameters);

                apertureRayParams->origin = g_hit;

                glm::vec3 a_c(0.0f);
                TransformComponent cameraTransform;
                auto apertureView = m_registry.view<CameraComponent>();
                for (auto ent: apertureView) {
                    auto entt = Entity(ent, this);
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
                auto apertureView = m_registry.view<CameraComponent>();
                for (auto ent: apertureView) {
                    auto entt = Entity(ent, this);
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
                auto apertureView = m_registry.view<CameraComponent>();
                for (auto ent: apertureView) {
                    auto entt = Entity(ent, this);
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
    }


    CameraComponent *Scene::getActiveCamera() {
        auto view = m_registry.view<CameraComponent>();
        CameraComponent *activeCamera = nullptr;
        // First pass: iterate through all cameras.
        // Keep updating activeCamera so that the last camera found with isActiveCamera() true wins.
        for (auto entityID: view) {
            Entity entity(entityID, this);
            auto &cameraComponent = entity.getComponent<CameraComponent>();
            if (cameraComponent.isActiveCamera()) {
                activeCamera = &cameraComponent;
            }
        }
        return activeCamera;
    }

    Entity Scene::getActiveCameraEntity() {
        Entity entity;
        auto view = m_registry.view<CameraComponent>();
        CameraComponent *activeCamera = nullptr;
        // First pass: iterate through all cameras.
        // Keep updating activeCamera so that the last camera found with isActiveCamera() true wins.
        for (auto entityID: view) {
            auto &cameraComponent = Entity(entityID, this).getComponent<CameraComponent>();
            if (cameraComponent.isActiveCamera()) {
                entity = Entity(entityID, this);
            }
        }
        return entity;
    }

    void Scene::deleteAllEntities() {
        auto view = m_registry.view<IDComponent>();
        for (auto entity: view) {
            // Wrap the registry entity in an Entity object for handling
            Entity e{entity, this};
            destroyEntity(e);
        }
    }

    Entity Scene::createEntityWithUUID(UUID uuid, const std::string &name) {
        Entity entity = {m_registry.create(), this};
        entity.addComponent<IDComponent>(uuid);
        entity.addComponent<TransformComponent>();
        auto &tag = entity.addComponent<TagComponent>();
        tag.Tag = name.empty() ? "Entity" : name;
        Log::Logger::getInstance()->info("Created Entity with UUID: {} and Tag: {}",
                                         entity.getUUID().operator std::string(), entity.getName());

        return entity;
    }


    Entity Scene::createEntity(const std::string &name) {
        return createEntityWithUUID(UUID(), name);
    }

    Entity Scene::getOrCreateEntityByName(const std::string &name) {
        // Check if the entity with the given UUID exists
        auto view = m_registry.view<TagComponent>();
        for (auto entityHandle: view) {
            auto &tagComponent = view.get<TagComponent>(entityHandle);
            if (tagComponent.getTag() == name) {
                // Entity with the given UUID already exists
                Entity existingEntity = {entityHandle, this};
                Log::Logger::getInstance()->trace("Retrieved existing Entity with UUID: {} and Tag: {}",
                                                  existingEntity.getUUID().operator std::string(),
                                                  existingEntity.getName());
                return existingEntity;
            }
        }
        // If not found, create a new entity with the given UUID and name
        return createEntity(name);
    }

    Entity Scene::getEntityByName(const std::string &name) {
        // Check if the entity with the given UUID exists
        auto view = m_registry.view<TagComponent>();
        for (auto entityHandle: view) {
            auto &tagComponent = view.get<TagComponent>(entityHandle);
            if (tagComponent.getTag() == name) {
                // Entity with the given UUID already exists
                Entity existingEntity = {entityHandle, this};
                Log::Logger::getInstance()->trace("Retrieved existing Entity with UUID: {} and Tag: {}",
                                                  existingEntity.getUUID().operator std::string(),
                                                  existingEntity.getName());
                return existingEntity;
            }
        }
        // If not found return empty entity
        return Entity();
    }

    void Scene::destroyEntity(Entity entity) {
        if (!entity) {
            Log::Logger::getInstance()->warning("Attempted to delete an entity that doesn't exist");
            return;
        }
        // Checking if the entity is still valid before attempting to delete
        if (m_registry.valid(entity)) {
            Log::Logger::getInstance()->info("Deleting Entity with UUID: {} and Tag: {}",
                                             entity.getUUID().operator std::string(), entity.getName());
            notifyComponentRemoval(entity);

            if (entity.hasComponent<ScriptableComponent>()) {
                entity.getComponent<ScriptableComponent>().instance->onDestroy();
            }
            // Perform the deletion
            m_registry.destroy(entity);
        } else {
            Log::Logger::getInstance()->warning(
                "Attempted to delete an invalid or already deleted entity");
        }
    }

    void Scene::destroyEntityRecursively(Entity entity) {
        // Delete children first
        if (entity.hasChildren()) {
            for (auto &child: entity.getChildren()) {
                destroyEntityRecursively(child);
            }
        }
        // Remove from parent
        if (entity.hasComponent<ParentComponent>()) {
            Entity parent = entity.getParent();
            parent.removeChild(entity);
        }
        // Destroy the entity
        m_context->activeScene()->destroyEntity(entity);
    }

    bool Scene::isDescendantOf(Entity entity, Entity potentialAncestor) {
        Entity currentParent = entity.getParent();
        while (currentParent) {
            if (currentParent == potentialAncestor)
                return true;
            currentParent = currentParent.getParent();
        }
        return false;
    }


    void Scene::notifyComponentRemoval(Entity entity) {
        // Check for each component type, and remove if the entity has the component
        if (entity.hasComponent<MeshComponent>()) {
            entity.removeComponent<MeshComponent>();
        }

        if (entity.hasComponent<MaterialComponent>()) {
            entity.removeComponent<MaterialComponent>();
        }

        if (entity.hasComponent<PointCloudComponent>()) {
            entity.removeComponent<PointCloudComponent>();
        }
        // Repeat for other components, adding more checks for each type of component
        // if (entity.hasComponent<OtherComponent>()) {
        //     entity.removeComponent<OtherComponent>();
        // }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }


    void Scene::notifyEditorsComponentUpdated(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, PointCloudComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, PointCloudComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }

    void Scene::notifyEditorsComponentRemoved(Entity entity, PointCloudComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }


    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER


    /** COMPONENT ADDED **/

    template<>
    void Scene::onComponentAdded<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template<>
    void Scene::onComponentAdded<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template<>

    void Scene::onComponentAdded<PointCloudComponent>(Entity entity, PointCloudComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }

    template<>
    void Scene::onComponentAdded<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ImageComponent>(Entity entity, ImageComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<GaussianComponent>(Entity entity, GaussianComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ParentComponent>(Entity entity, ParentComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ChildrenComponent>(Entity entity, ChildrenComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<GroupComponent>(Entity entity, GroupComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<VisibleComponent>(Entity entity, VisibleComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TemporaryComponent>(Entity entity, TemporaryComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS &component) {
    }

    template<>
    void Scene::onComponentAdded<LightSourceComponent>(Entity entity, LightSourceComponent &component) {
    }
    template<>
    void Scene::onComponentAdded<QuadricCollectionComponent>(Entity entity, QuadricCollectionComponent &component) {
    }
    template<>
    void Scene::onComponentAdded<ScriptableComponent>(Entity entity, ScriptableComponent &component) {
    }

    /** COMPONENT REMOVE **/

    template<>
    void Scene::onComponentRemoved<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template<>
    void Scene::onComponentRemoved<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template<>
    void Scene::onComponentRemoved<PointCloudComponent>(Entity entity, PointCloudComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }

    template<>
    void Scene::onComponentRemoved<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ImageComponent>(Entity entity, ImageComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<GaussianComponent>(Entity entity, GaussianComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ParentComponent>(Entity entity, ParentComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ChildrenComponent>(Entity entity, ChildrenComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<GroupComponent>(Entity entity, GroupComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<VisibleComponent>(Entity entity, VisibleComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TemporaryComponent>(Entity entity, TemporaryComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS &component) {
    }

    template<>
    void Scene::onComponentRemoved<LightSourceComponent>(Entity entity, LightSourceComponent &component) {
    }
    template<>
    void Scene::onComponentRemoved<QuadricCollectionComponent>(Entity entity, QuadricCollectionComponent &component) {
    }
    template<>
    void Scene::onComponentRemoved<ScriptableComponent>(Entity entity, ScriptableComponent &component) {
    }

    /** COMPONENT UPDATE **/
    template<>
    void Scene::onComponentUpdated<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<PointCloudComponent>(Entity entity, PointCloudComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ImageComponent>(Entity entity, ImageComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<GaussianComponent>(Entity entity, GaussianComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ParentComponent>(Entity entity, ParentComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ChildrenComponent>(Entity entity, ChildrenComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<GroupComponent>(Entity entity, GroupComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<VisibleComponent>(Entity entity, VisibleComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TemporaryComponent>(Entity entity, TemporaryComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<GaussianComponent2DGS>(Entity entity, GaussianComponent2DGS &component) {
    }

    template
    <>
    void Scene::onComponentUpdated<LightSourceComponent>(Entity entity, LightSourceComponent &component) {
    }

    template
    <>
    void Scene::onComponentUpdated<QuadricCollectionComponent>(Entity entity, QuadricCollectionComponent &component) {
    }
    template
    <>
    void Scene::onComponentUpdated<ScriptableComponent>(Entity entity, ScriptableComponent &component) {
    }

    DISABLE_WARNING_POP
}
