//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_PROPERTIESLAYER
#define MULTISENSE_VIEWER_PROPERTIESLAYER

#include <Viewer/Assets/Gaussian2DAssetLoader.h>

#include "Viewer/Rendering/ImGui/Layer.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/ImGui/LayerUtils.h"

namespace VkRender {
    class PropertiesLayer : public Layer {
    public:
        void onAttach() override;

        void onDetach() override;

        void onUIRender() override;

        void onFinishedRender() override;

        void setScene(std::weak_ptr<Scene> scene) override;

        void reconstructionTab();

    public:
        template<typename T, typename UIFunction>
        void drawComponent(const std::string &name, Entity entity, UIFunction uiFunction);

        void drawComponents(Entity entity);

        void objectProperties();

        void drawRendererSettingsTab();

        bool m_tmp = false; // TODO remove
        bool m_visibility = true;
        float noiseSlider = 0.1f;

        Entity m_selectionContext;
        std::future<LayerUtils::LoadFileInfo> m_loadFileFuture;
        std::future<LayerUtils::LoadFileInfo> m_loadFolderFuture;
        char m_tagBuffer[256]; // Adjust size as needed
        bool m_needsTagUpdate = true;

        template<typename T>
        void displayAddComponentEntry(const std::string &entryName);

        void handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo &loadFileInfo);

        void checkFileImportCompletion();

        void checkFolderImportCompletion();

        static bool
        drawVec3Control(const std::string &label, glm::vec3 &values, float resetValue, float columnWidth, float speed);

        static bool drawFloatControl(const std::string &label, float &value, float resetValue, float speed,
                                     float columnWidth);

        bool
        drawVec2Control(const std::string &label, glm::vec2 &values, float resetValue, float speed, float columnWidth);

        bool
        drawQuatControl(const std::string &label, glm::quat &quat, float resetValue = 0.0f,
                        float speed = 1.0f, float columnWidth = 100.0f);
    };

    static void updateScene2DGS(Entity selectionCtx, std::shared_ptr<Gaussian2DAsset> gaussianAsset,
                                std::shared_ptr<Scene> scene) {
        // Generate Entities from PointCloudAsset
        for (int i = 0; i < gaussianAsset->numPoints; ++i) {
            std::string quadricName =
                    "2DGS " + std::to_string(i) + ":" + selectionCtx.getName();
            auto entityInstance = scene->getOrCreateEntityByName(quadricName);
            entityInstance.setParent(selectionCtx);
            auto &temp = entityInstance.getOrAddComponent<TemporaryComponent>();

            // Get or create TransformComponent and set position and rotation.
            auto &transform = entityInstance.getOrAddComponent<TransformComponent>();
            transform.setPosition(gaussianAsset->positions[i]);
            transform.setRotationQuaternion(gaussianAsset->rotations[i]);
            //transform.setScale({gaussianAsset->scale_x[i], gaussianAsset->scale_y[i], 0.0f});

            // Apply parent's transformation.
            glm::mat4 parentMatrix = selectionCtx.getComponent<TransformComponent>().
                    getTransform();
            glm::mat4 worldMatrix = parentMatrix * transform.getTransform();
            transform.setTransform(worldMatrix);
            // Setup MeshComponent with quadric parameters.
            auto &mesh = entityInstance.getOrAddComponent<MeshComponent>(GAUSSIAN_2D);
            mesh.polygonMode() = VK_POLYGON_MODE_FILL;
            auto meshParameters = std::dynamic_pointer_cast<Gaussian2DMeshParameters>(
                mesh.meshParameters);
            // Setup MaterialComponent.
            auto &material = entityInstance.getOrAddComponent<MaterialComponent>();
            material.useTexture = false;
            material.diffuse = 0.3f;
            material.specular = 0.7f;
            material.phongExponent = 128.0f;
            material.fragmentShaderName = "NoMaterial.frag";
            meshParameters->color = gaussianAsset->colors[i];
            meshParameters->opacity = gaussianAsset->opacity[i];
            meshParameters->covX = gaussianAsset->scale_x[i];
            meshParameters->covY = gaussianAsset->scale_y[i];
            material.alphaMode = AlphaMode::Blend;
        }
    }
}

#endif //MULTISENSE_VIEWER_PROPERTIESLAYER
