//
// Created by magnus on 8/14/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DLAYER_H
#define MULTISENSE_VIEWER_EDITOR3DLAYER_H


#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"
#include "Viewer/VkRender/Editors/EditorDefinitions.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class Editor3DLayer : public Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        /** Called once per frame **/
        void onUIRender() override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(5.0f, 55.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width - 5.0f,
                                        m_editor->ui()->height - 55.0f); // Size (width, height)


            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("Editor3DLayer", nullptr, window_flags);



            /*
            static bool toggle = false;
            ImGui::Checkbox("Save image toggle", &toggle);
            if (toggle) {
                m_editor->ui()->saveRenderToFile = true;
            }
            */

            /*
            auto view = m_scene->getRegistry().view<CameraComponent, TagComponent>();
            static int selectedCameraIndex = 0;
            static std::string currentCameraName = "Select camera";

            ImGui::SetNextItemWidth(200.0f);
            if (ImGui::BeginCombo("combo 1", currentCameraName.c_str(), 0)) {
                int index = 0;
                for (auto entity : view) {
                    auto& tag = view.get<TagComponent>(entity);
                    std::string cameraName = tag.Tag; // Assuming the CameraComponent has a 'name' field.
                    bool is_selected = (selectedCameraIndex == index);
                    if (ImGui::Selectable(cameraName.c_str(), is_selected)) {
                        selectedCameraIndex = index;
                        currentCameraName = cameraName;
                        m_editor->ui()->shared->m_selectedEntity = Entity(entity, m_scene.get());
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }

                    ++index;
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Preview selected camera entity", &m_editor->ui()->setActiveCamera);

            ImGui::Checkbox("Render depth", &m_editor->ui()->renderDepth);
            m_editor->ui()->saveRenderToFile = ImGui::Button("Save image");

            if (ImGui::Button("Create camera")) {
                auto scene = m_context->activeScene();
                auto entity = scene->createNewCamera("NewCamera", 1280, 720);
                auto &transform = entity.getComponent<TransformComponent>();
                auto &camera = entity.getComponent<CameraComponent>();
                entity.addComponent<MeshComponent>(1);
                transform.setPosition(m_editor->ui()->editorCamera->pose.pos);
                auto quaternion = glm::quat_cast(m_editor->ui()->editorCamera->getFlyCameraTransMat());
                transform.setQuaternion(quaternion);
                camera().pose.pos = m_editor->ui()->editorCamera->pose.pos;
                camera().pose.q = quaternion;
                camera().updateViewMatrix();
            };
 */

            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach()
        override {

        }
    };

}

#endif //MULTISENSE_VIEWER_EDITOR3DLAYER_H
