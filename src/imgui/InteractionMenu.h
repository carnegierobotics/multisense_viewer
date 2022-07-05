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
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f));

        ImGui::Begin("InteractionMenu", &pOpen, window_flags);
        int i = 0;

        ImGui::BeginChild("blah");


        /*
        float my_tex_w = 100;
        float my_tex_h = 100;
        ImGui::PushID("Preview Button");
        ImVec2 frame_padding = ImVec2(0,0);                     // Size of the image we want to make visible
        ImVec2 size = ImVec2(my_tex_w, my_tex_h);                     // Size of the image we want to make visible
        ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
        ImVec2 uv1 = ImVec2(1.0f, 1.0f);// UV coordinates for (32,32) in our texture
        ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);         // no background
        ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
        bool clicked = ImGui::ImageButton(handles->info->id, size, uv0, uv1, 0, bg_col, tint_col);
        clicked =  ImGui::IsItemActive();
        ImGui::PopID();
*/
        int imageButtonHeight = 100;
        static int pressed_count[3] = {0, 0, 0};
        const char* labels[3] = {"Preview Device", "Device Information", "Configure Device"};
        for (int i = 0; i < 3; i++) {
            float offset = 150;
            ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + (i * offset),
                                       (handles->info->height / 2) - (imageButtonHeight / 2)));

            ImGui::PushID(i);
            ImVec2 size = ImVec2(100.0f, 100.0f);                     // Size of the image we want to make visible
            ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
            ImVec2 uv1 = ImVec2(1.0f, 1.0f);

            ImVec4 bg_col = ImVec4(0.054, 0.137, 0.231, 1.0f);         // Match bg color
            ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
            if (ImGui::ImageButton(handles->info->imageButtonTextureDescriptor[i], size, uv0, uv1, 0, bg_col, tint_col))
                pressed_count[i] += 1;
            ImGui::PopID();

            ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + (i * offset),
                                       (handles->info->height / 2) + (imageButtonHeight / 2) + 8));

            ImGui::Text("%s", labels[i]);

            ImGui::SameLine();
        }
        ImGui::NewLine();

        if (handles->devices != nullptr) {


        }

        ImGui::EndChild();

        ImGui::ShowDemoWindow();
        ImGui::End();
        ImGui::PopStyleColor(); // bg color

        if (pressed_count[0] == 0)
            return;
        //// Start of configuration layout
        //bool pOpen = true;
        //ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height / 4));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (handles->devices != nullptr) {
            for (auto &d: *handles->devices) {

                // Create dropdown
                if (d.state == ArActiveState) {
                    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyResizeDown;
                    if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                        if (ImGui::BeginTabItem("Streaming")) {
                            addStreamingTab(handles, &d);

                            ImGui::EndTabItem();
                        }
                        if (ImGui::BeginTabItem("Configuration")) {
                            addConfigurationTab(handles, &d);

                            ImGui::EndTabItem();
                        }
                        ImGui::EndTabBar();
                    }

                }
            }
        }

        ImGui::ShowDemoWindow();

        ImGui::End();
    }

    void addConfigurationTab(GuiObjectHandles *handles, Element *d) {
        ImGui::Text("This tab is reserved for configurations. Placeholders, not implemented");

        ImGui::SliderFloat("Exposure time", &handles->sliderOne, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("LED duty cycle", &handles->sliderTwo, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);

        ImGui::SliderFloat("X", &handles->sliderOne, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("Y", &handles->sliderTwo, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("Z", &handles->sliderThree, -5.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);

    }

    void addStreamingTab(GuiObjectHandles *handles, Element *d) {

        if (ImGui::BeginCombo("Resolution",
                              d->selectedStreamingMode.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto &mode: d->modes) {

                bool is_selected = (mode.modeName ==
                                    d->selectedStreamingMode); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(mode.modeName.c_str(), is_selected)) {
                    d->selectedStreamingMode = mode.modeName;

                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
            }
            ImGui::EndCombo();
        } // End dropdown

        ImGui::Checkbox("Depth Image", &d->depthImage);
        ImGui::Checkbox("Color Image", &d->colorImage);
        ImGui::Checkbox("Point Cloud", &d->pointCloud);

        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x, ImGui::GetCursorPos().y + 10.0f));

        d->button = ImGui::Button("Preview Bar", ImVec2(175.0f, 20.0f));


    }


};

#endif //MULTISENSE_INTERACTIONMENU_H
