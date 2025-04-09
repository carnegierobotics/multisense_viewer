//
// Created by magnus on 2/24/25.
//

#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRendererLayer.h"
#include "EditorDifferentiableRenderer.h"
#include "EditorDifferentiableRendererLayerUI.h"

namespace VkRender{

        /** Called once upon this object creation**/
        void EditorDifferentiableRendererLayer::onAttach()
        {
        }

        /** Called after frame has finished rendered **/
        void EditorDifferentiableRendererLayer::onFinishedRender()
        {
        }


        /** Called once per frame **/
        void EditorDifferentiableRendererLayer::onUIRender()
        {
            // Set window position and size
            // Set window position and size
            ImVec2 window_pos = ImVec2(m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width - window_pos.x,
                                        m_editor->ui()->height - window_pos.y); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            // Create the parent window
            ImGui::Begin("EditorPathTracerLayer", nullptr, window_flags);

            auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_editor->ui());

            imageUI->reloadRenderer |= ImGui::Button("Reload");
            ImGui::SameLine();

            imageUI->step = ImGui::Button("Step");
            ImGui::SameLine();

            ImGui::Checkbox("Step##Toggle", &imageUI->toggleStep);
            ImGui::SameLine(0.0f, 10.0f);

            ImGui::Checkbox("Save Debug Info##Toggle", &imageUI->saveDebugInfo);
            ImGui::SameLine();
            ImGui::SameLine(); // Dropdown for selecting render kernel
            const char* selections[] = {"CPU", "GPU"}; // TODO This should come from selectSyclDevices
            ImGui::SetNextItemWidth(100.0f);
            ImGui::Combo("##Select Device Type", &imageUI->selectedDeviceIndex, selections,IM_ARRAYSIZE(selections));
            imageUI->kernelDevice = selections[imageUI->selectedDeviceIndex];

            auto* editor =  reinterpret_cast<EditorDifferentiableRenderer *>(m_editor);
            ImGui::Text("Iteration: %d", editor->m_stepIteration);
            ImGui::End();
        }

        /** Called once upon this object destruction **/
        void EditorDifferentiableRendererLayer::onDetach()
        {
        }

}