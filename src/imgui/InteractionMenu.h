//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include "Layer.h"

class InteractionMenu : public Layer {
public:

// Create global object for convenience in other functions

    void onFinishedRender() override {

    }

    void OnUIRender(GuiObjectHandles *handles) override {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height / 2));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);

        if (handles->devices != nullptr) {
            for (auto &d: *handles->devices) {

                // Create dropdown
                if (d.state == ArActiveState) {

                    if (ImGui::BeginCombo("Resolution",
                                          d.selectedStreamingMode.c_str())) // The second parameter is the label previewed before opening the combo.
                    {
                        for (auto &mode: d.modes) {

                            bool is_selected = (mode.modeName ==
                                                d.selectedStreamingMode); // You can store your selection however you want, outside or inside your objects
                            if (ImGui::Selectable(mode.modeName.c_str(), is_selected)) {
                                d.selectedStreamingMode = mode.modeName;

                            }
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
                        }
                        ImGui::EndCombo();
                    } // End dropdown

                    ImGui::Checkbox("Depth Image", & d.depthImage);
                    ImGui::Checkbox("Color Image", & d.colorImage);
                    ImGui::Checkbox("Point Cloud", & d.pointCloud);

                    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x, ImGui::GetCursorPos().y + 50.0f));

                    d.btnShowPreviewBar = ImGui::Button("Preview Bar", ImVec2(175.0f, 40.0f));


                }
            }
        }
        ImGui::End();
    }


};

#endif //MULTISENSE_INTERACTIONMENU_H
