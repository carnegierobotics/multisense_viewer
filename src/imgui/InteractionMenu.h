//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include "Layer.h"
#include "imgui_user.h"

class InteractionMenu : public AR::Layer {
public:

// Create global object for convenience in other functions

    void onFinishedRender() override {

    }


    void OnUIRender(AR::GuiObjectHandles *handles) override {
        if (handles->devices->empty()) return;
        bool allUnavailable = true;
        for (auto &d: *handles->devices) {
            if (d.state == AR_STATE_ACTIVE)
                allUnavailable = false;
        }

        if (allUnavailable)
            return;

        // Check if stream was interrupted by a disconnect event and reset pages events across all devices
        for (auto &d: *handles->devices) {
            for (auto &s: d.streams)
                if (s.second.playbackStatus == AR_PREVIEW_RESET) {
                    s.second.playbackStatus = AR_PREVIEW_NONE;
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
                ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + ((float) i * offset),
                                           (handles->info->height / 2) - ((float) imageButtonHeight / 2)));

                ImGui::PushID(i);
                ImVec2 size = ImVec2(100.0f,
                                     100.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = ImVec4(0.054, 0.137, 0.231, 1.0f);         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                if (ImGui::ImageButton(handles->info->imageButtonTextureDescriptor[i], size, uv0, uv1, 0, bg_col,
                                       tint_col))
                    page[i] = true;
                ImGui::PopID();

                ImGui::SetCursorPos(ImVec2(handles->info->sidebarWidth + ((float) i * offset),
                                           (handles->info->height / 2) + ((float) imageButtonHeight / 2) + 8));

                ImGui::Text("%s", labels[i]);

                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::ShowDemoWindow();
            ImGui::End();
            ImGui::PopStyleColor(); // bg color
        }
    }


    void
    addDropDown(AR::GuiObjectHandles *handles, AR::StreamingModes *stream) {

        if (stream->modes.empty() || stream->sources.empty())
            return;

        StreamIndex streamIndex = stream->streamIndex;
        std::string id = stream->name;

        ImVec2 position = ImGui::GetCursorScreenPos();
        position.x += (handles->info->controlAreaWidth / 2) - (handles->info->controlDropDownWidth / 2);
        ImVec2 pos = position;
        pos.x += handles->info->controlDropDownWidth;
        pos.y += handles->info->controlDropDownHeight;


        if (openDropDown[streamIndex]) {
            pos = position;
            if (animationLength[streamIndex] < handles->info->dropDownHeightOpen)
                animationLength[streamIndex] += 1000 * handles->info->frameTimer;
            pos.y += animationLength[streamIndex];
            pos.x += handles->info->controlDropDownWidth;
        } else {
            animationLength[streamIndex] = 0;
        }

        ImGui::GetWindowDrawList()->AddRectFilled(position, pos, ImColor(0.17, 0.157, 0.271, 1.0f), 10.0f, 0);

        ImGui::SetCursorScreenPos(position);
        ImGui::CustomSelectable(id.c_str(), &openDropDown[streamIndex], 0,
                                ImVec2(handles->info->controlDropDownWidth, handles->info->controlDropDownHeight));

        if (openDropDown[streamIndex]) {

            ImVec2 sourceComboPos((handles->info->controlAreaWidth / 2) - (handles->info->dropDownWidth / 2),
                                  ImGui::GetCursorPosY());
            ImGui::SetCursorPos(sourceComboPos);
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::Text("Data source:");

            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::SetNextItemWidth(handles->info->dropDownWidth);
            if (ImGui::BeginCombo(std::string("##Source" + std::to_string(streamIndex)).c_str(),
                                  stream->sources[stream->selectedSourceIndex].c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (int n = 0; n < stream->sources.size(); n++) {

                    const bool is_selected = (stream->selectedSourceIndex == n);
                    if (ImGui::Selectable(stream->sources[n].c_str(), is_selected)) {
                        stream->selectedSourceIndex = n;
                        stream->selectedStreamingSource = stream->sources[stream->selectedSourceIndex];

                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }


            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::Text("Resolution:");


            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::SetNextItemWidth(handles->info->dropDownWidth);
            if (ImGui::BeginCombo(std::string("##Resolution" + std::to_string(streamIndex)).c_str(),
                                  stream->modes[stream->selectedModeIndex].c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (int n = 0; n < stream->modes.size(); n++) {
                    const bool is_selected = (stream->selectedModeIndex == n);
                    if (ImGui::Selectable(stream->modes[n].c_str(), is_selected)) {
                        stream->selectedModeIndex = n;
                        stream->selectedStreamingMode = stream->modes[stream->selectedModeIndex];

                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::Dummy(ImVec2(0.0f, 20.0f));

            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            std::string btnLabel = "Start##" + std::to_string(streamIndex);
            if (ImGui::Button(btnLabel.c_str())) {
                stream->playbackStatus = AR_PREVIEW_PLAYING;
                Log::Logger::getInstance()->info("Pressed Play for preview {}", id.c_str());
            }
            ImGui::SameLine();
            btnLabel = "Pause##" + std::to_string(streamIndex);
            if (ImGui::Button(btnLabel.c_str())) {
                stream->playbackStatus = AR_PREVIEW_PAUSED;
                Log::Logger::getInstance()->info("Pressed Pause for preview {}", id.c_str());
            }
            ImGui::SameLine();
            btnLabel = "Stop##" + std::to_string(streamIndex);
            if (ImGui::Button(btnLabel.c_str())) {
                stream->playbackStatus = AR_PREVIEW_NONE;
                Log::Logger::getInstance()->info("Pressed Stop for preview {}", id.c_str());
                firstSetup[stream->streamingOrder] = true; // TODO index correctly compared to which preview is open
            }

            ImGui::Dummy(ImVec2(0.0f, 40.0f));


            //openDropDownMenu(handles, position);
        }
    }

private:
    bool page[PAGE_TOTAL_PAGES] = {false, false, false};
    bool drawActionPage = true;
    bool openDropDown[AR_PREVIEW_TOTAL_MODES + 1] = {false};
    float animationLength[AR_PREVIEW_TOTAL_MODES + 1] = {false};
    bool firstSetup[9] = {true, true, true, true, true, true, true, true, true};

    void buildDeviceInformation(AR::GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.054, 0.137, 0.231, 1.0f)); // TODO USE named colors
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_DEVICE_INFORMATION] = false;
            drawActionPage = true;
        }

        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }

    void buildPreview(AR::GuiObjectHandles *handles) {
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

        for (auto &d: *handles->devices) {
            if (d.cameraName == "Virtual Camera") {
                addStreamPlaybackControls(AR_PREVIEW_VIRTUAL_LEFT, "Virtual preview", &d);
            }
        }
        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }


    void buildConfigurationPreview(AR::GuiObjectHandles *handles) {
        for (auto &dev: *handles->devices) {
            if (dev.state != AR_STATE_ACTIVE)
                continue;

            // Control page
            ImGui::BeginGroup();
            createControlArea(handles, dev);
            ImGui::EndGroup();

            // Viewing page
            ImGui::BeginGroup();
            createViewingArea(handles, dev);
            ImGui::EndGroup();

        }
        //ImGui::PopStyleColor(); // bg color
        //ImGui::End();
    }

    void createViewingArea(AR::GuiObjectHandles *handles, AR::Element &dev) {

        /** CREATE VIEWING PREVIEWS **/
        uint32_t previewWindowCount = 0;
        for (auto &d: *handles->devices) {
            if (d.state == AR_STATE_ACTIVE && d.selectedPreviewTab == TAB_2D_PREVIEW) {
                for (auto &stream: d.streams) {
                    if (stream.second.playbackStatus == AR_PREVIEW_PLAYING) {
                        stream.second.streamingOrder = previewWindowCount;
                        ImGui::PushStyleColor(ImGuiCol_WindowBg,
                                              ImVec4(0.034, 0.107, 0.201, 0.05f)); // TODO use named colors
                        createPreviewArea(handles, stream.second.streamIndex, previewWindowCount, d);
                        previewWindowCount++;
                        ImGui::PopStyleColor(); // Bg color
                    }
                }
            }
        }

        /** CREATE TOP NAVIGATION BAR **/
        bool pOpen = true;
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse;;

        ImGui::SetNextWindowPos(
                ImVec2(handles->info->sidebarWidth + handles->info->controlAreaWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.034, 0.107, 0.201, 1.0f));

        handles->info->viewingAreaWidth =
                handles->info->width - handles->info->sidebarWidth - handles->info->controlAreaWidth;

        ImGui::SetNextWindowSize(ImVec2(handles->info->viewingAreaWidth, handles->info->viewingAreaHeight),
                                 ImGuiCond_Always);
        ImGui::Begin("ViewingArea", &pOpen, window_flags);


        ImGui::Dummy(ImVec2((handles->info->viewingAreaWidth / 2) - 30.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PushFont(handles->info->font18);
        ImGui::Text("Viewing");
        ImGui::PopFont();

        ImGui::Dummy(ImVec2((handles->info->viewingAreaWidth / 2) - 80.0f, 0.0f));
        ImGui::SameLine();

        if (dev.selectedPreviewTab == TAB_2D_PREVIEW)
            ImGui::PushStyleColor(ImGuiCol_Button, AR::green);
        else
            ImGui::PushStyleColor(ImGuiCol_Button, AR::red);

        if (ImGui::Button("2D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = TAB_2D_PREVIEW;
            Log::Logger::getInstance()->info("Profile {}: 2D preview pressed", dev.name.c_str());
        }
        ImGui::PopStyleColor();

        if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD)
            ImGui::PushStyleColor(ImGuiCol_Button, AR::green);
        else
            ImGui::PushStyleColor(ImGuiCol_Button, AR::red);

        ImGui::SameLine();
        if (ImGui::Button("3D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = TAB_3D_POINT_CLOUD;
            Log::Logger::getInstance()->info("Profile {}: 3D preview pressed", dev.name.c_str());
        }
        ImGui::PopStyleColor();
        ImGui::End();

        ImGui::PopStyleColor(); // Bg color



    }


    void createPreviewArea(AR::GuiObjectHandles *handles, StreamIndex streamIndex, uint32_t i, AR::Element d) {
        float viewAreaElementPosX = handles->info->sidebarWidth + handles->info->controlAreaWidth + 40.0f;

        //TODO remove hardcoded positions and sizes from this function
        if (firstSetup[i]) {
            handles->info->viewAreaElementPositionsY[i] =
                    75.0f + ((float) i * (320.0f + ((float) handles->info->height * 0.13f)));
            firstSetup[i] = false;
            Log::Logger::getInstance()->info("Created preview for {} in order {}", (uint32_t) streamIndex, i);

        } else
            handles->info->viewAreaElementPositionsY[i] -= handles->mouseBtns.wheel * 20.0f;

        ImGui::SetNextWindowPos(ImVec2(viewAreaElementPosX, handles->info->viewAreaElementPositionsY[i]),
                                ImGuiCond_Always); // TODO REMOVE HARDCODED VALUE

        handles->info->viewAreaElementSizeY = 250.0f + ((float) handles->info->height * 0.10f);

        ImGui::SetNextWindowSize(ImVec2(handles->info->viewingAreaWidth - 80.0f, handles->info->viewAreaElementSizeY),
                                 ImGuiCond_Always);  // TODO REMOVE HARDCODED VALUES
        static bool open = true;
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus;
        std::string windowName = std::to_string(streamIndex);
        ImGui::Begin((std::string("View Area##") + windowName).c_str(), &open, window_flags);


        ImGui::Text("%s", d.streams.find(streamIndex)->second.name.c_str());
        ImGui::SameLine();
        std::string info1 = "| Monochrome##";
        ImGui::Text("%s", (info1 + std::to_string((int) streamIndex)).c_str());
        ImGui::SameLine();
        std::string info2 = "| FPS##";
        ImGui::Text("%s", (info2 + std::to_string((int) streamIndex)).c_str());


        ImGui::End();
    }

    void createControlArea(AR::GuiObjectHandles *handles, AR::Element &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.034, 0.107, 0.201, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(handles->info->controlAreaWidth, handles->info->controlAreaHeight));
        ImGui::Begin("ControlArea", &pOpen, window_flags);

        for (auto &d: *handles->devices) {
            // Create dropdown
            if (d.state == AR_STATE_ACTIVE) {

                ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Control")) {

                        if (ImGui::Button("Back")) {
                            page[PAGE_CONFIGURE_DEVICE] = false;
                            drawActionPage = true;
                        }

                        // Create Control Area
                        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, handles->info->tabAreaHeight),
                                                ImGuiCond_Always);
                        ImGui::BeginChild("Dropdown Area", ImVec2(handles->info->controlAreaWidth,
                                                                  handles->info->height - handles->info->tabAreaHeight),
                                          false, ImGuiCond_Always);

                        for (auto &d: *handles->devices) {
                            if (d.state == AR_STATE_ACTIVE) {
                                // TODO DRAW DROPDOWNS BASED ON MODES FOUND IN CAMERACONNECTION.CPP during initialization
                                for (int i = 0; i < AR_PREVIEW_TOTAL_MODES + 1; i++) {
                                    if (d.streams.find(i) != d.streams.end())
                                        addDropDown(handles, &d.streams[i]);
                                }
                                /*  addDropDown(handles, "1. Camera", AR_PREVIEW_LEFT, &d.streams[AR_PREVIEW_VIRTUAL_LEFT]);
                                  addDropDown(handles, "2. Point Cloud", AR_PREVIEW_POINT_CLOUD_VIRTUAL,
                                              &d.streams[AR_PREVIEW_POINT_CLOUD_VIRTUAL]);

                                  addDropDown(handles, "1. Left Camera", AR_PREVIEW_LEFT, &d.streams[AR_PREVIEW_LEFT]);
                                  addDropDown(handles, "2. Right Camera", AR_PREVIEW_RIGHT, &d.streams[AR_PREVIEW_RIGHT]);
                                  addDropDown(handles, "3. Auxiliary Camera", AR_PREVIEW_AUXILIARY,
                                              &d.streams[AR_PREVIEW_AUXILIARY]);
                                  addDropDown(handles, "4. Disparity", AR_PREVIEW_DISPARITY,
                                              &d.streams[AR_PREVIEW_DISPARITY]);

                                  addDropDown(handles, "5. Point Cloud", AR_PREVIEW_POINT_CLOUD,
                                              &d.streams[AR_PREVIEW_POINT_CLOUD]);
  */

                            }
                        }


                        ImGui::EndChild();

                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Configuration")) {

                        bool active = false;

                        active |= ImGui::Checkbox("Auto Exposure", &d.parameters.ep.autoExposure);
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        if (!d.parameters.ep.autoExposure){
                            active |= ImGui::SliderInt("Exposure", reinterpret_cast<int *>(&d.parameters.ep.exposure), 10, 30000);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        } else {
                            active |= ImGui::SliderInt("Auto exposure max value", reinterpret_cast<int *>(&d.parameters.ep.autoExposureMax), 10, 35000);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            active |= ImGui::SliderInt("Auto exposure decay", reinterpret_cast<int *>(&d.parameters.ep.autoExposureDecay), 1, 100);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            active |= ImGui::SliderFloat("Auto exposure intensity",  &d.parameters.ep.autoExposureTargetIntensity, 0, 1);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            active |= ImGui::SliderFloat("Auto exposure threshold", &d.parameters.ep.autoExposureThresh, 0, 1);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        }

                        ImGui::Dummy(ImVec2(0.0f, 20.0f));
                        active |= ImGui::Checkbox("Auto White Balance", &d.parameters.wb.autoWhiteBalance);
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        if (!d.parameters.wb.autoWhiteBalance){
                            active |= ImGui::SliderFloat("White balance red", &d.parameters.wb.whiteBalanceRed, 1, 100);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            active |= ImGui::SliderFloat("White balance blue", &d.parameters.wb.whiteBalanceBlue, 1, 100);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        } else {
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            active |= ImGui::SliderFloat("Auto white balance threshold", &d.parameters.wb.autoWhiteBalanceThresh, 1, 100);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            active |= ImGui::SliderInt("Auto white balance decay",
                                                       reinterpret_cast<int *>(&d.parameters.wb.autoWhiteBalanceDecay), 1, 100);
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        }

                        active |= ImGui::SliderFloat("Stereo Post Filter Strength", &d.parameters.stereoPostFilterStrength, 0, 1);
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        active |= ImGui::SliderFloat("Gamma", &d.parameters.gamma, 0, 10);
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        active |= ImGui::SliderFloat("Gain", &d.parameters.gain, 0, 10);
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        active |= ImGui::SliderFloat("FPS", &d.parameters.fps, 0, 60);
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));

                        ImGui::Dummy(ImVec2(0.0f, handles->info->height - ImGui::GetCursorScreenPos().y));

                        if (active) {
                            Log::Logger::getInstance()->info("A config was pressed");
                        }

                        d.parameters.update = active;

                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Future Tab 1")) {

                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Future Tab 2")) {

                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
        }
        ImGui::PopStyleColor();
        ImGui::End();
        ImGui::PopStyleVar(2);


    }


    void addStreamPlaybackControls(CameraStreamInfoFlag streamIndex, std::string label, AR::Element *d) {
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
            if (stream.playbackStatus != AR_PREVIEW_PLAYING)
                stream.playbackStatus = AR_PREVIEW_PLAYING;
            else
                stream.playbackStatus = AR_PREVIEW_STOPPED;
        }

        ImGui::EndGroup();
    }


};

#endif //MULTISENSE_INTERACTIONMENU_H
