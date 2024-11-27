//
// Created by magnus on 8/1/24.
//

#ifndef MULTISENSE_VIEWER_MAINCONTEXTLAYER_H
#define MULTISENSE_VIEWER_MAINCONTEXTLAYER_H

#include "Viewer/Rendering/ImGui/Layer.h"
#include "Viewer/Rendering/Editors/MultiSenseViewer/SidebarEditor/AddDevicePopup.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class MainContextLayer : public VkRender::Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }



        /** Called once per frame **/
        void onUIRender() override {

            float menuBarHeight = 25.0f;

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, menuBarHeight); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width,
                                       m_editor->ui()->height - window_pos.y); // Size (width, height)


            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoFocusOnAppearing |
                    ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

            // Create the parent window
            ImGui::Begin("EditorBorderWindow", nullptr, window_flags);

            ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, menuBarHeight));

            /*
            if (m_editor->ui()->shared->openAddDevicePopup) {
                ImGui::OpenPopup("add_device_modal");
            }
            */

            addPopup(m_context, m_editor);
            ImGui::End();

            ImGui::PopStyleVar(2);


        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}

#endif //MULTISENSE_VIEWER_MAINCONTEXTLAYER_H
