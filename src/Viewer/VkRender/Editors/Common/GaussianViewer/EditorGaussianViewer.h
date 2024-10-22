//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
#define MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/VkRender/RenderResources/SyclGaussianGFX.h"
#include "Viewer/VkRender/Core/SyclDeviceSelector.h"

namespace VkRender {

    class EditorGaussianViewer : public Editor {
    public:
        EditorGaussianViewer() = delete;

        explicit EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid = UUID());


        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onUpdate() override;

        ~EditorGaussianViewer() override {

        }

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;
        void onEditorResize() override;
        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        void onKeyCallback(const Input& input) override;

    private:
        std::shared_ptr<Camera> m_activeCamera;
        std::shared_ptr<Scene> m_activeScene;

        SyclDeviceSelector m_deviceSelector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::GPU);
        SyclGaussianGFX m_syclGaussianGfx;


    };
}

#endif //MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
