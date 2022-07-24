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
        //if (handles->devices->empty()) return;

        // Check if stream was interrupted by a disconnect event and reset pages events across all devices
        for (auto &d: *handles->devices) {
            for (auto &s: d.streams)
                if (s.second.playbackStatus == PREVIEW_RESET) {
                    s.second.playbackStatus = PREVIEW_NONE;
                    std::fill_n(page, PAGE_TOTAL_PAGES, false);
                    drawActionPage = true;
                }
        }

        for (int i = 0; i < PAGE_TOTAL_PAGES; ++i) {
            if (page[i]) {
                if (drawActionPage) {
                    drawActionPage = false;
                }
                switch (i) {
                    case PAGE_PREVIEW_DEVICES:
                        buildPreview(handles);
                        break;
                    case PAGE_DEVICE_INFORMATION:
                        buildDeviceInformation(handles);
                        break;
                    case PAGE_CONFIGURE_DEVICE:
                        buildConfigurationPreview(handles);
                        break;
                }
            }
        }

        if (drawActionPage) {
            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
            ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f));
            ImGui::Begin("InteractionMenu", &pOpen, window_flags);

            int imageButtonHeight = 100;
            const char *labels[3] = {"Preview Device", "Device Information", "Configure Device"};
            for (int i = 0; i < PAGE_TOTAL_PAGES; i++) {
                float offset = 150;
                ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + (i * offset),
                                           (handles->info->height / 2) - (imageButtonHeight / 2)));

                ImGui::PushID(i);
                ImVec2 size = ImVec2(100.0f, 100.0f);                     // Size of the image we want to make visible
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = ImVec4(0.054, 0.137, 0.231, 1.0f);         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                if (ImGui::ImageButton(handles->info->imageButtonTextureDescriptor[i], size, uv0, uv1, 0, bg_col,
                                       tint_col))
                    page[i] = true;
                ImGui::PopID();

                ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + (i * offset),
                                           (handles->info->height / 2) + (imageButtonHeight / 2) + 8));

                ImGui::Text("%s", labels[i]);

                ImGui::SameLine();
            }

            /*
            bool open = true;
            // Create Control Area
            ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(440.0f, 720.0f));
            ImGui::Begin("Control Area", &open, ImGuiCond_Always);

            ImVec2 position = ImGui::GetCursorScreenPos();
            //bool clicked(ImGui::Button("Custom", ImVec2(200.0f, 30.0f)));
            ImVec2 pos = position;
            pos.y += 30.0f;
            pos.x += 200.0f;

            ImGui::GetWindowDrawList()->AddRectFilled(position, pos, ImColor(0.17, 0.157, 0.271, 1.0f), 10.0f, 0);
            bool clicked = ImGui::InvisibleButton("Item#1", ImVec2(200.0f, 30.0f));

            if (clicked && openDropDown) {
                openDropDown = false;
                length = 0;
            } else if (clicked) openDropDown = true;

            if (openDropDown) {
                printf("Clicked\n");
                ImVec2 position = ImGui::GetCursorScreenPos();

                ImVec2 pos = position;
                if (length < 250)
                    length += 1;

                pos.y += length;
                pos.x += 150;
                ImGui::SetNextWindowPos(position, 0);

                ImGui::BeginChild("Test1", ImVec2(300, 300));
                ImGui::GetWindowDrawList()->AddRectFilled(position, pos, ImColor(0.17, 0.157, 0.271, 1.0f), 10.0f, 0);


                ImGui::Text("Select data source");

                const char *items[] = {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ",
                                       "KKKK", "LLLLLLL", "MMMM", "OOOOOOO"};
                static int item_current_idx = 0; // Here we store our selection data as an index.
                const char *combo_preview_value = items[item_current_idx];  // Pass in the preview value visible before opening the combo (it could be anything)

                if (ImGui::BeginCombo("Custom", combo_preview_value, 0)) {
                    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {

                        const bool is_selected = (item_current_idx == n);
                        if (ImGui::Selectable(items[n], is_selected)) {
                            item_current_idx = n;

                        }
                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }


                ImGui::EndChild();

                //openDropDownMenu(handles, position);
            }
            ImGui::End();

             */

            /*
            const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK", "LLLLLLL", "MMMM", "OOOOOOO" };
            static int item_current_idx = 0; // Here we store our selection data as an index.
            const char* combo_preview_value = items[item_current_idx];  // Pass in the preview value visible before opening the combo (it could be anything)

            if (ImGui::CustomCombo("Custom", ImVec2(200.0f, 30.0f), &openDropDown, combo_preview_value, 0)) {
                for (int n = 0; n < IM_ARRAYSIZE(items); n++) {

                    const bool is_selected = (item_current_idx == n);
                    if (ImGui::Selectable(items[n], is_selected)) {
                        item_current_idx = n;

                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            */

            ImGui::NewLine();
            ImGui::ShowDemoWindow();
            ImGui::PopStyleColor(); // bg color
            ImGui::End();
        }

    }

private:
    bool page[PAGE_TOTAL_PAGES] = {false, false, false};
    bool drawActionPage = true;
    bool openDropDown = false;
    float length = 0.0f;

    void openDropDownMenu(GuiObjectHandles *handles, ImVec2 position) {

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.17, 0.157, 0.271, 1.0f));
        position.y += 30.0f;
        position.x += 200.0f;
        ImGui::SetNextWindowPos(position, 0);
        ImGui::BeginChild("child", ImVec2(200.0f, 150.0f));

        ImGui::Text("Select data source");

        const char *items[] = {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ",
                               "KKKK", "LLLLLLL", "MMMM", "OOOOOOO"};
        static int item_current_idx = 0; // Here we store our selection data as an index.
        const char *combo_preview_value = items[item_current_idx];  // Pass in the preview value visible before opening the combo (it could be anything)

        if (ImGui::BeginCombo("Custom", combo_preview_value, 0)) {
            for (int n = 0; n < IM_ARRAYSIZE(items); n++) {

                const bool is_selected = (item_current_idx == n);
                if (ImGui::Selectable(items[n], is_selected)) {
                    item_current_idx = n;

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Text("Select resolution");

        const char *items2[] = {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ",
                                "KKKK", "LLLLLLL", "MMMM", "OOOOOOO"};
        static int item_current_idx2 = 0; // Here we store our selection data as an index.
        const char *combo_preview_value2 = items2[item_current_idx2];  // Pass in the preview value visible before opening the combo (it could be anything)

        if (ImGui::BeginCombo("Custom2", combo_preview_value2, 0)) {
            for (int n = 0; n < IM_ARRAYSIZE(items2); n++) {

                const bool is_selected = (item_current_idx2 == n);
                if (ImGui::Selectable(items2[n], is_selected)) {
                    item_current_idx2 = n;

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }


        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    void buildDeviceInformation(GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_DEVICE_INFORMATION] = false;
            drawActionPage = true;
        }

        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }

    void buildPreview(GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 0.0f));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_PREVIEW_DEVICES] = false;
            drawActionPage = true;
        }

        ImVec2 pos = ImGui::GetCursorPos();
        pos.y += 5;
        pos.x += 50;
        ImGui::SetCursorPos(pos);

        for (auto& d: *handles->devices) {
            if (d.cameraName == "Virtual Camera") {
                addStreamPlaybackControls(PREVIEW_VIRTUAL, "Virtual preview", &d);
            }
        }
        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }


    void buildConfigurationPreview(GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);

        // Control page
        ImGui::BeginGroup();
        createControlArea(handles);
        ImGui::EndGroup();

        // Viewing page


        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }

    void createControlArea(GuiObjectHandles *handles) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth + handles->info->offset5px, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.034, 0.107, 0.201, 1.0f));
        ImGui::BeginChild("Viewing", ImVec2(handles->info->controlAreaWidth, handles->info->controlAreaHeight), false,
                          window_flags);

        for (auto &d: *handles->devices) {
            // Create dropdown
            if (d.state == ArActiveState) {
                ImGuiTabBarFlags tab_bar_flags = 0;//ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                    if (ImGui::BeginTabItem("Control")) {

                        if (ImGui::Button("Back")) {
                            page[PAGE_CONFIGURE_DEVICE] = false;
                            drawActionPage = true;
                        }

                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Configuration")) {

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Future Tab 1")) {

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Future Tab 2")) {

                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
        }
        ImGui::PopStyleColor();
        ImGui::EndChild();
    }

    /*
    void buildConfigurationPreview(GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse |
                       ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 0.0f));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);

        for (auto &d: *handles->devices) {
            // Create dropdown
            if (d.state == ArActiveState) {
                ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                    if (ImGui::BeginTabItem("Streaming")) {
                        if (ImGui::Button("Back")) {
                            page[PAGE_CONFIGURE_DEVICE] = false;
                            drawActionPage = true;
                        }

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


        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, handles->info->height / 3), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.035, 0.078, 0.125, 0.10f));
        ImGui::BeginChild("ConfigurationPreview",
                          ImVec2(handles->info->width - handles->info->sidebarWidth, 2 * handles->info->height / 3),
                          false, window_flags);

        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + 5, ImGui::GetCursorPosY()));
        // Create Preview Window
        for (auto &d: *handles->devices) {
            // Create dropdown
            if (d.state == ArActiveState) {
                ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("2D Preview", tab_bar_flags)) {
                    if (ImGui::BeginTabItem("Streaming")) {
                        d.selectedPreviewTab = TAB_2D_PREVIEW;

                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("3D Point cloud")) {
                        d.selectedPreviewTab = TAB_3D_POINTCLOUD;

                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
        }

        ImGui::EndChild();
        ImGui::PopStyleColor(); // child window bg color

        ImGui::NewLine();
        ImGui::PopStyleColor(); // main bg color
        ImGui::End();
    }

    void addConfigurationTab(GuiObjectHandles *handles, Element *d) {
        ImGui::Text("This tab is reserved for configurations. Placeholders, not implemented");

        if (ImGui::Button("Back")) {
            page[PAGE_CONFIGURE_DEVICE] = false;
            drawActionPage = true;
        }

        ImGui::SliderFloat("Exposure time", &handles->sliderOne, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("LED duty cycle", &handles->sliderTwo, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);

        ImGui::SliderFloat("X", &handles->sliderOne, -4.0f, 4.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("Y", &handles->sliderTwo, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("Z", &handles->sliderThree, -5.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);

    }

    void addStreamingTab(GuiObjectHandles *handles, Element *d) {


        ImVec2 pos = ImGui::GetCursorPos();
        pos.y += 5;
        pos.x += 50;
        ImGui::SetCursorPos(pos);

        if (d->cameraName == "Virtual Camera") {
            addStreamPlaybackControls(PREVIEW_VIRTUAL, "Virtual preview", d);

        } else {
            addStreamPlaybackControls(PREVIEW_LEFT, "Left preview", d);
            ImGui::SameLine();
            addStreamPlaybackControls(PREVIEW_RIGHT, "Right preview", d);
            ImGui::SameLine();
            addStreamPlaybackControls(PREVIEW_DISPARITY, "Disparity preview", d);
            ImGui::SameLine();
            addStreamPlaybackControls(PREVIEW_AUXILIARY, "Auxiliary preview", d);
        }

    }

     */
    void addStreamPlaybackControls(CameraStreamInfoFlag streamIndex, std::string label, Element *d) {
        ImGui::BeginGroup();
        ImGui::Text("%s", label.c_str());
        auto &stream = d->streams[streamIndex];
        ImGui::SetNextItemWidth(200);
        const char *combo_preview_value = stream.sources[stream.selectedSourceIndex].c_str();  // Pass in the preview value visible before opening the combo (it could be anything)
        std::string srcLabel = "##Source" + std::to_string(streamIndex);
        if (ImGui::BeginCombo(srcLabel.c_str(), combo_preview_value,
                              ImGuiComboFlags_HeightSmall)) {
            for (int n = 0; n < stream.sources.size(); n++) {

                const bool is_selected = (stream.selectedSourceIndex == n);
                if (ImGui::Selectable(stream.sources[n].c_str(), is_selected)) {
                    stream.selectedSourceIndex = n;
                    stream.selectedStreamingSource = stream.sources[stream.selectedSourceIndex];

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SetNextItemWidth(200);

        std::string resLabel = "##Resolution" + std::to_string(streamIndex);
        if (ImGui::BeginCombo(resLabel.c_str(), stream.modes[stream.selectedModeIndex].c_str(),
                              ImGuiComboFlags_HeightSmall)) {
            for (int n = 0; n < stream.modes.size(); n++) {
                const bool is_selected = (stream.selectedModeIndex == n);
                if (ImGui::Selectable(stream.modes[n].c_str(), is_selected)) {
                    stream.selectedModeIndex = n;
                    stream.selectedStreamingMode = stream.modes[stream.selectedModeIndex];

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        std::string btnLabel = "Play/Pause##" + std::to_string(streamIndex);
        if (ImGui::Button(btnLabel.c_str())) {
            if (stream.playbackStatus != PREVIEW_PLAYING)
                stream.playbackStatus = PREVIEW_PLAYING;
            else
                stream.playbackStatus = PREVIEW_STOPPED;
        }

        ImGui::EndGroup();
    }


};

#endif //MULTISENSE_INTERACTIONMENU_H
