//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "Emitter.h"
#include "Helpers.h"

#include <Viewer/Rendering/Components/GaussianComponent.h>
#include <Viewer/Rendering/Core/KeyInput.h>

#include "imgui.h"


namespace VkRender {
    void Emitter::onUpdate(Timestep ts) {
        auto scene = m_entity.getScene();

        auto gsEntity = scene->getEntityByName("2DGS");
        glm::vec3 position(0.0f);
        glm::vec3 direction(0.0f);
        if (gsEntity) {
            for (int i = 0; auto &gs: gsEntity.getComponent<GaussianComponent2DGS>().emissions) {
                if (gs > 0.0f) {
                    position = gsEntity.getComponent<GaussianComponent2DGS>().positions[i];
                }
                ++i;
            }
        }

        auto &mesh = getComponent<MeshComponent>();
        auto cylinder = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
        if (cylinder) {
            direction = glm::normalize(cylinder->direction);


        }

        // Select a Quadric
        if (Input::isKeyPressed(GLFW_KEY_RIGHT)) {
            quadricIndex++;
        }
        if (Input::isKeyPressed(GLFW_KEY_LEFT)) {
            quadricIndex--;
        }

        auto quadricCollection = scene->getEntityByName("QuadricCollection");
        if (quadricCollection && quadricCollection.hasChildren()) {
            auto children = quadricCollection.getChildren();
            // Clamp or check the index
            if (quadricIndex < 0 || quadricIndex >= children.size()) {
                // Handle the invalid index: reset, clamp, or simply ignore the change.
                // For example, clamp the index:
                quadricIndex = std::clamp(quadricIndex, 0, static_cast<int>(children.size() - 1));
            }
            auto quadricEntity = children[quadricIndex];

            if (quadricEntity) {
                auto &mesh = quadricEntity.getComponent<MeshComponent>();
                auto &transform = quadricEntity.getComponent<TransformComponent>();
                auto quadric = std::dynamic_pointer_cast<QuadricMeshParameters>(mesh.meshParameters);

                PathTracer::QuadricInputAssembly quad;
                quad.a = quadric->a;
                quad.b = quadric->b;
                quad.c = quadric->c;
                quad.t_x = quadric->t_x;
                quad.t_y = quadric->t_y;
                quad.transform = transform;

                float beta = 0.0f;
                if (RayHelpers::computeWorldHitPoint(position, direction, quad, hitPosition, hitNormal, beta)) {
                    cylinder->setMagnitude(glm::length(hitPosition - position));
                    cylinder->setOrigin(position);
                } else {
                    hitPosition = {-99.0f, 0.0f, 0.0f};
                    cylinder->setOrigin(hitPosition);

                }



            }
        }
    }

    void Emitter::onDestroy() {
    }

    void Emitter::onCreate() {
        auto scene = m_entity.getScene();
    }
}
