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

    struct Editor3DViewportUI : public EditorUI {
        bool showVideoControlPanel = false;
        bool resetPlayback = false;
        bool stopCollectingRenderCommands = false;
        // Constructor that copies everything from base EditorUI
        explicit Editor3DViewportUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

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
            ImVec2 window_pos = ImVec2( m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width - window_pos.x,
                                        m_editor->ui()->height - window_pos.y); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("Editor3DLayer", nullptr, window_flags);
            auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_editor->ui());


            ImGui::Checkbox("Stop Collecting RenderCommands", &imageUI->stopCollectingRenderCommands); ImGui::SameLine();

            ImGui::Checkbox("Video control panel", &imageUI->showVideoControlPanel); ImGui::SameLine();
            if (imageUI->showVideoControlPanel) {
                imageUI->resetPlayback = ImGui::Button("Reset"); ImGui::SameLine();
            }
            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach()
        override {

        }
    };

}

#endif //MULTISENSE_VIEWER_EDITOR3DLAYER_H
