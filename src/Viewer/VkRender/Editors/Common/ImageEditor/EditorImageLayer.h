//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGERLAYER
#define MULTISENSE_VIEWER_EDITORIMAGERLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"

namespace VkRender {


    class EditorImageLayer : public Layer {

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
            // Set window position and size
            ImVec2 window_pos = ImVec2(50.0f, 5.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor.editorUi->width - window_pos.x,
                                        m_editor.editorUi->height - window_pos.y); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            // Create the parent window
            ImGui::Begin("EditorImageLayer", nullptr, window_flags);

            ImGui::SetNextItemWidth(150.0f);
            // If we have a device connected:
            if (m_context->multiSense()->anyMultiSenseDeviceOnline()) {

                // Available sources from camera
                auto profile& = m_context->multiSense()->getSelectedMultiSenseProfile();
                std::vector<std::string> &availableSources = profile.deviceData.sources;

                if (availableSources.empty()) {
                    availableSources.emplace_back("Error: No sources available");
                }

                std::string& previewValue = availableSources.front();

                if (profile.deviceData.enabledSources.contains(*reinterpret_cast<int *>(&m_editor))){
                    previewValue = profile.deviceData.enabledSources[*reinterpret_cast<int *>(&m_editor)];
                }

                if (ImGui::BeginCombo(("Select source:" + std::to_string(*reinterpret_cast<int *>(&m_editor))).c_str(), previewValue.c_str(), ImGuiComboFlags_HeightLarge)) {
                    for (int n = 0; n < availableSources.size(); n++) {
                        bool is_selected = (profile.deviceData.enabledSources[*reinterpret_cast<int *>(&m_editor)] == availableSources[n]);
                        if (ImGui::Selectable(availableSources[n].c_str(), is_selected)) {
                            profile.deviceData.enabledSources[*reinterpret_cast<int *>(&m_editor)] = availableSources[n];
                        }

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
                    }
                    ImGui::EndCombo();
                }

            }
            ImGui::End();


        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}
#endif //MULTISENSE_VIEWER_EDITORIMAGERLAYER
