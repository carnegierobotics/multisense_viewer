//
// Created by magnus on 2/4/25.
//

#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayer.h"

#include <Viewer/Rendering/ImGui/IconsFontAwesome6.h>
#include <Viewer/Rendering/ImGui/LayerUtils.h>

#include "EditorPathTracer.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"
#include "Viewer/Rendering/Editors/Editor.h"

namespace VkRender {
    /** Called once upon this object creation**/
    void EditorPathTracerLayer::onAttach() {
    }


    /** Called once per frame **/
    void EditorPathTracerLayer::onUIRender() {
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

        static int currentTab = -1; // keep outside the loop
        const VerticalIconTab kTabs[] =
        {
            //{ICON_FA_TV, "Renderer Settings", [this] { drawRendererSettingsTab(); }, 300.0f},
            {ICON_FA_BUG, "Debug", [this] {drawDebugViewTab(); }, 300.0f},
        };
        LayerUtils::drawVerticalIconLeftPopupTabs(kTabs, IM_ARRAYSIZE(kTabs), currentTab, m_editor);

        ImGui::End();
    }

    void EditorPathTracerLayer::drawDebugViewTab() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_editor->ui());
        ImGui::Text("Show BVH:"); ImGui::SameLine();

        ImGui::Checkbox("##Show BVH", &imageUI->showBVH);
        if (imageUI->showBVH) {
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("TLAS:"); ImGui::SameLine(); ImGui::Checkbox("##Show TLAS", &imageUI->showTLAS);
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("BLAS:"); ImGui::SameLine(); ImGui::Checkbox("##Show BLAS", &imageUI->showBLAS);
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("Min Level:"); ImGui::SameLine();
            ImGui::SliderInt("##BVH Depth Min", &imageUI->bvhLevelMin, 0, 30);
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("Max Level:"); ImGui::SameLine();
            ImGui::SliderInt("##BVH Depth Max", &imageUI->bvhLevelMax, 1, 30);
            ImGui::Spacing(); ImGui::SameLine();

        }

    }


    /** Called once upon this object destruction **/
    void EditorPathTracerLayer::onDetach() {
    }

    void EditorPathTracerLayer::onFinishedRender() {
    }
}
