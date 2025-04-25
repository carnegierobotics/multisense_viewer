//
// Created by mgjer on 25/04/2025.
//


#include "Viewer/Rendering/Editors/3DViewport/Editor3DLayer.h"

#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Editors/3DViewport/Editor3DViewport.h"

#include <ImGuizmo.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace VkRender {
    void Editor3DLayer::onAttach() {
    }

    void Editor3DLayer::onFinishedRender() {
    }

    void Editor3DLayer::onUIRender() {
        // Set window position and size
        ImVec2 window_pos = ImVec2(m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
        ImVec2 window_size = ImVec2(m_editor->ui()->width - window_pos.x,
                                    m_editor->ui()->height - window_pos.y); // Size (width, height)

        // Set window flags to remove decorations
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground |
                ImGuiWindowFlags_NoBringToFrontOnFocus;

        // Set next window position and size
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
        // Create the parent window
        ImGui::Begin("Editor3DLayer", nullptr, window_flags);

        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_editor->ui());
        auto editor = reinterpret_cast<Editor3DViewport *>(m_editor);

        ImGui::Checkbox("Active camera", &imageUI->renderFromViewpoint);
        ImGui::SameLine();
        imageUI->saveNextFrame = ImGui::Button("Save");
        ImGui::SameLine();
        imageUI->reloadViewportShader = ImGui::Button("Reload Shader");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        if (ImGui::BeginCombo("Image Type",
                              imageUI->selectedImageType == OutputTextureImageType::Color ? "Color" : "Depth")) {
            if (ImGui::Selectable("Color", imageUI->selectedImageType == OutputTextureImageType::Color)) {
                if (imageUI->selectedImageType != OutputTextureImageType::Color) {
                    imageUI->selectedImageType = OutputTextureImageType::Color;
                    editor->onRenderSettingsChanged();
                }
            }
            if (ImGui::Selectable("Depth", imageUI->selectedImageType == OutputTextureImageType::Depth)) {
                if (imageUI->selectedImageType != OutputTextureImageType::Depth) {
                    imageUI->selectedImageType = OutputTextureImageType::Depth;
                    editor->onRenderSettingsChanged();
                    imageUI->depthColorOption = DepthColorOption::Invert;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();

        // Show Depth options if Depth is selected
        if (imageUI->selectedImageType == OutputTextureImageType::Depth) {
            ImGui::Text("Depth Color Options");
            ImGui::SameLine();

            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::BeginCombo("Color Option",
                                  imageUI->depthColorOption == DepthColorOption::None
                                      ? "None"
                                      : imageUI->depthColorOption == DepthColorOption::Invert
                                            ? "Invert"
                                            : imageUI->depthColorOption == DepthColorOption::Normalize
                                                  ? "Normalize"
                                                  : imageUI->depthColorOption == DepthColorOption::JetColormap
                                                        ? "Colormap (Jet)"
                                                        : "Colormap (Viridis)")) {
                if (ImGui::Selectable("Invert", imageUI->depthColorOption == DepthColorOption::Invert)) {
                    if (imageUI->depthColorOption != DepthColorOption::Invert) {
                        imageUI->depthColorOption = DepthColorOption::Invert;
                        editor->onRenderSettingsChanged();
                    }
                }
                if (ImGui::Selectable("Normalize", imageUI->depthColorOption == DepthColorOption::Normalize)) {
                    if (imageUI->depthColorOption != DepthColorOption::Normalize) {
                        imageUI->depthColorOption = DepthColorOption::Normalize;
                        editor->onRenderSettingsChanged();
                    }
                }
                if (ImGui::Selectable("Colormap (Jet)",
                                      imageUI->depthColorOption == DepthColorOption::JetColormap)) {
                    if (imageUI->depthColorOption != DepthColorOption::JetColormap) {
                        imageUI->depthColorOption = DepthColorOption::JetColormap;
                        editor->onRenderSettingsChanged();
                    }
                }
                if (ImGui::Selectable("Colormap (Viridis)",
                                      imageUI->depthColorOption == DepthColorOption::ViridisColormap)) {
                    if (imageUI->depthColorOption != DepthColorOption::ViridisColormap) {
                        imageUI->depthColorOption = DepthColorOption::ViridisColormap;
                        editor->onRenderSettingsChanged();
                    }
                }
                ImGui::EndCombo();
            }
        } else {
            imageUI->depthColorOption = DepthColorOption::None;
        }

        static int selection = 0;
        if (Input::isKeyClicked(GLFW_KEY_Q)) {
            selection = -1;
        }
        if (Input::isKeyClicked(GLFW_KEY_W)) {
            selection = ImGuizmo::TRANSLATE;
        }
        if (Input::isKeyClicked(GLFW_KEY_E)) {
            selection = ImGuizmo::ROTATE;
        }
        if (Input::isKeyClicked(GLFW_KEY_R)) {
            selection = ImGuizmo::SCALE;
        }

        // view gizmo
        auto scene = m_context->activeScene();
        auto entity = m_context->getSelectedEntity();
        if (selection >= 0 && entity && entity.hasComponent<TransformComponent>()) {
            auto &transformComponent = entity.getComponent<TransformComponent>();
            auto camera = editor->getCamera();
            auto matrices = camera->matrices;
            float *viewPtr = glm::value_ptr(matrices.view);
            float *projectionPtr = glm::value_ptr(matrices.projection);
            static glm::mat4 matrix(1.0f);
            static glm::mat4 deltaMatrix(1.0f);

            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist();
            ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, ImGui::GetWindowWidth(),
                              ImGui::GetWindowHeight());
            glm::mat4 transform = transformComponent.getTransform();
            float snap = 0;
            ImGuizmo::Manipulate(viewPtr, projectionPtr, static_cast<ImGuizmo::OPERATION>(selection), ImGuizmo::WORLD,
                                 glm::value_ptr(transform));

            if (ImGuizmo::IsUsing()) {
                glm::vec3 translation, rotation, scale;
                decomposeTransform(transform, translation, rotation, scale);
                transformComponent.setPosition(translation);
                transformComponent.setRotationEuler(rotation);
                transformComponent.setScale(scale);
                editor->ui()->occludedByGizmo = true;
            } else {
                editor->ui()->occludedByGizmo = false;
            }
        }
        ImGui::End();
    }

    void Editor3DLayer::onDetach() {
    }
}
