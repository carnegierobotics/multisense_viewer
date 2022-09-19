//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include "Layer.h"
#include "imgui_user.h"
#include "GLFW/glfw3.h"

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
            if (d.state == AR_STATE_RESET) {
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

            ImGui::PushStyleColor(ImGuiCol_WindowBg, AR::CRLCoolGray);
            ImGui::Begin("InteractionMenu", &pOpen, window_flags);

            int imageButtonHeight = 100;
            const char *labels[3] = {"Preview Device", "Device Information", "Configure Device"};
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

            for (int i = 0; i < PAGE_TOTAL_PAGES; i++) {
                float imageSpacing = 150.0f;

                // width of menu buttons layout. Needed to draw on center of screen.
                float xOffset = ((handles->info->width - handles->info->sidebarWidth) / 2) -
                                (((float) imageSpacing * (float) PAGE_CONFIGURE_DEVICE) +
                                 ((float) PAGE_TOTAL_PAGES * 100.0f) / 2) +
                                handles->info->sidebarWidth +
                                ((float) i * imageSpacing);

                ImGui::SetCursorPos(ImVec2(xOffset,

                                           (handles->info->height / 2) - ((float) imageButtonHeight / 2)));

                ImVec2 size = ImVec2(100.0f,
                                     100.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = AR::CRLCoolGray;         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                if (ImGui::ImageButton(labels[i], handles->info->imageButtonTextureDescriptor[i], size, uv0, uv1,
                                       bg_col, tint_col))
                    page[i] = true;

                ImGui::SetCursorPos(ImVec2(xOffset, (handles->info->height / 2) + ((float) imageButtonHeight / 2) + 8));

                ImGui::Text("%s", labels[i]);

                ImGui::SameLine();
            }

            ImGui::PopStyleVar();
            ImGui::NewLine();
            //ImGui::ShowDemoWindow();
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

        bool highlight = false;
        if (stream->playbackStatus == AR_PREVIEW_PLAYING)
            highlight = true;

        ImGui::CustomSelectable(id.c_str(), &openDropDown[streamIndex], highlight, 0,
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

            /*
            ImGui::Dummy(ImVec2(sourceComboPos.x - 30.0f, 0.0f));
            ImGui::SameLine();
            ImGui::HelpMarker("\n  HelpMarker  \n\n");
            ImGui::SameLine();
             */


            ImGui::SetCursorPos(ImVec2(sourceComboPos.x, ImGui::GetCursorPosY()));
            ImGui::SetNextItemWidth(handles->info->dropDownWidth);
            if (ImGui::BeginCombo(std::string("##Resolution" + std::to_string(stream->streamIndex)).c_str(),
                                  stream->modes[stream->selectedModeIndex].c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (int n = 0; n < stream->modes.size(); n++) {
                    const bool is_selected = (stream->selectedModeIndex == n);
                    if (ImGui::Selectable(stream->modes[n].c_str(), is_selected)) {
                        stream->selectedModeIndex = n;
                        stream->selectedStreamingMode = Utils::stringToCameraResolution(
                                stream->modes[stream->selectedModeIndex]);
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
            }

            if (streamIndex == AR_PREVIEW_POINT_CLOUD) {
                ImGui::SameLine(0, 150.0f); // TODO Hardcoded positioning values
                ImGui::HelpMarker(
                        "  Why is my point cloud black?\n   Solution: start 'Luma Rectified Left' data source in 1. Left Sensor  \n\n");
            }

            ImGui::Dummy(ImVec2(0.0f, 40.0f));
        }
    }

private:
    bool page[PAGE_TOTAL_PAGES] = {false, false, false};
    bool drawActionPage = true;
    bool openDropDown[AR_PREVIEW_TOTAL_MODES + 1] = {false};
    float animationLength[AR_PREVIEW_TOTAL_MODES + 1] = {false};


    void buildDeviceInformation(AR::GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, AR::CRLCoolGray); // TODO USE named colors
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

        ImGui::PushStyleColor(ImGuiCol_WindowBg, AR::CRLCoolGray);
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_PREVIEW_DEVICES] = false;
            drawActionPage = true;
        }

        /*
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
        */

        ImGui::PopStyleColor(); // bg color
        ImGui::End();

    }


    void buildConfigurationPreview(AR::GuiObjectHandles *handles) {
        ImGui::ShowDemoWindow();

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
    }

    void createViewingArea(AR::GuiObjectHandles *handles, AR::Element &dev) {

        /** CREATE TOP NAVIGATION BAR **/
        bool pOpen = true;
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse;;

        ImGui::SetNextWindowPos(
                ImVec2(handles->info->sidebarWidth + handles->info->controlAreaWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, AR::CRLCoolGray);

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

/*
    void createPreviewArea(AR::GuiObjectHandles *handles, StreamIndex streamIndex, uint32_t i, AR::Element d) {
        // Calculate window position
        //float viewAreaElementPosX = handles->info->sidebarWidth + handles->info->controlAreaWidth + 40.0f;
        float viewAreaElementPosX = (((handles->info->viewingAreaWidth / 2.0f) + handles->info->sidebarWidth +
                                      handles->info->controlAreaWidth - ((0.25f * handles->info->width) / 2.0f)) -
                                     handles->info->previewBorderPadding);


        // The top left corner of the ImGui window that encapsulates the quad with the texture playing.
        // Equation is a (to-do)

        handles->info->hoverState = ImGui::IsWindowHovered("ControlArea", ImGuiHoveredFlags_AnyWindow);
        if (!handles->info->hoverState && !handles->input->getButton(GLFW_KEY_LEFT_CONTROL)) {
            handles->accumulatedActiveScroll -= ImGui::GetIO().MouseWheel * 100.0f;
        }

        float viewAreaElementPosY = (handles->info->height * 0.25f) + handles->accumulatedActiveScroll +
                                    ((float) i * (275.0f * (handles->info->width / 1280.0f)));

        handles->info->viewAreaElementPositionsY[i] = viewAreaElementPosY;

        ImGui::SetNextWindowPos(ImVec2(viewAreaElementPosX, handles->info->viewAreaElementPositionsY[i]),
                                ImGuiCond_Always);

        // Calculate window size
        handles->info->viewAreaElementSizeX =
                0.25f * handles->info->width + (handles->info->previewBorderPadding * 2);

        handles->info->viewAreaElementSizeY =
                ((handles->info->previewBorderPadding) + 0.25f * 720.0f * (handles->info->width / 1280.0f)) * 1.11f;


        ImGui::SetNextWindowSize(ImVec2(handles->info->viewAreaElementSizeX, handles->info->viewAreaElementSizeY),
                                 ImGuiCond_Always);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        static bool open = true;
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus;
        std::string windowName = std::to_string(streamIndex);
        ImGui::Begin((std::string("View Area##") + windowName).c_str(), &open, window_flags);

        // The colored bars around the preview window is made up of rectangles
        // Top bar
        ImVec2 topBarRectMin(viewAreaElementPosX, handles->info->viewAreaElementPositionsY[i]);
        ImVec2 topBarRectMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                             handles->info->viewAreaElementPositionsY[i] + (handles->info->previewBorderPadding / 2));
        ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMin, topBarRectMax, ImColor(0.11, 0.215, 0.33, 1.0f), 0.0f,
                                                  0);

        // extension of top bar in different color
        ImVec2 topBarRectMinExtended(viewAreaElementPosX, topBarRectMax.y);
        ImVec2 topBarRectMaxExtended(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                                     topBarRectMax.y + 5.0f);

        ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMinExtended, topBarRectMaxExtended,
                                                  ImColor(0.21, 0.283, 0.36, 1.0f), 0.0f,
                                                  0);

        // Left bar

        ImVec2 leftBarMin(viewAreaElementPosX, topBarRectMaxExtended.y);
        ImVec2 leftBarMax(viewAreaElementPosX + handles->info->previewBorderPadding,
                          topBarRectMaxExtended.y + handles->info->viewAreaElementSizeY -
                          (handles->info->previewBorderPadding * 0.5f));

        ImGui::GetWindowDrawList()->AddRectFilled(leftBarMin, leftBarMax, ImColor(0.21, 0.283, 0.36, 1.0f), 0.0f,
                                                  0);

        // Right bar
        ImVec2 rightBarMin(
                viewAreaElementPosX + handles->info->viewAreaElementSizeX - handles->info->previewBorderPadding,
                topBarRectMaxExtended.y);
        ImVec2 rightBarMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                           topBarRectMaxExtended.y + handles->info->viewAreaElementSizeY -
                           (handles->info->previewBorderPadding * 0.5f));

        ImGui::GetWindowDrawList()->AddRectFilled(rightBarMin, rightBarMax, ImColor(0.21, 0.283, 0.36, 1.0f), 0.0f,
                                                  0);

        // Bottom bar
        ImVec2 bottomBarMin(viewAreaElementPosX, topBarRectMaxExtended.y + handles->info->viewAreaElementSizeY -
                                                 (handles->info->previewBorderPadding) - 13.0f);
        ImVec2 bottomBarMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                            topBarRectMaxExtended.y + handles->info->viewAreaElementSizeY -
                            (handles->info->previewBorderPadding * 0.5));

        ImGui::GetWindowDrawList()->AddRectFilled(bottomBarMin, bottomBarMax, ImColor(0.21, 0.283, 0.36, 1.0f), 0.0f,
                                                  0);

        // Text
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::Dummy(ImVec2(20.0f, 0.0f));
        ImGui::SameLine();
        ImGui::Text("%s", d.streams.find(streamIndex)->second.name.c_str());
        ImGui::SameLine(0, 210.0f * (handles->info->width / 1280));
        ImGui::PushID(streamIndex);
        ImGui::Text("Fps: %.2f", d.parameters.fps);
        ImGui::PopID();
        ImGui::End();
        ImGui::PopStyleVar(); // Window padding
    }
*/
    void createDoubleWindowPreview(AR::GuiObjectHandles *handles, AR::Element &dev) {
        AR::GuiLayerUpdateInfo *info = handles->info;
        info->viewingAreaWidth = info->width - info->sidebarWidth - info->controlAreaWidth;

        // The top left corner of the ImGui window that encapsulates the quad with the texture playing.
        // Equation is a (to-do)

        info->hoverState = ImGui::IsWindowHovered("ControlArea", ImGuiHoveredFlags_AnyWindow);
        if (!info->hoverState && !handles->input->getButton(GLFW_KEY_LEFT_CONTROL)) {
            handles->accumulatedActiveScroll -= ImGui::GetIO().MouseWheel * 100.0f;
        }

        int cols, rows;
        switch (dev.layout) {
            case PREVIEW_LAYOUT_NONE:
                break;
            case PREVIEW_LAYOUT_SINGLE:
                cols = 1;
                rows = 1;
                break;
            case PREVIEW_LAYOUT_DOUBLE:
                rows = 2;
                cols = 1;
                break;
            case PREVIEW_LAYOUT_DOUBLE_SIDE_BY_SIDE:
                rows = 1;
                cols = 2;
                break;
            case PREVIEW_LAYOUT_QUAD:
                cols = 2;
                rows = 2;
                break;
            case PREVIEW_LAYOUT_NINE:
                cols = 3;
                rows = 3;
                break;
        }

        int index = 0;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {

                float newWidth = (info->width - (640 + (5.0f + (5.0f * (float) cols)))) / (float) cols;
                float newHeight = newWidth * (10.0f / 16.0f); // aspect ratio 16:10 of camera images
                // Calculate window size
                info->viewAreaElementSizeX = newWidth;
                info->viewAreaElementSizeY = newHeight;

                float offsetX;
                if (dev.layout == PREVIEW_LAYOUT_DOUBLE || dev.layout == PREVIEW_LAYOUT_SINGLE) {
                    offsetX = (info->controlAreaWidth + info->sidebarWidth + ((info->viewingAreaWidth - newWidth) / 2));
                }
                else
                    offsetX = (info->controlAreaWidth + info->sidebarWidth + 5.0f);

                float viewAreaElementPosX = offsetX + ((float) col * (newWidth + 10.0f));


                ImGui::SetNextWindowSize(ImVec2(info->viewAreaElementSizeX, info->viewAreaElementSizeY),
                                         ImGuiCond_Always);

                dev.row[index] = (float) row;
                // Calculate window position
                float viewAreaElementPosY=
                        info->tabAreaHeight + ((float) row * (info->viewAreaElementSizeY + 10.0f));

                ImGui::SetNextWindowPos(ImVec2(viewAreaElementPosX, viewAreaElementPosY),
                                        ImGuiCond_Always);

                static bool open = true;
                ImGuiWindowFlags window_flags =
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse;

                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                ImGui::Begin((std::string("View Area") + std::to_string(index)).c_str(), &open, window_flags);

                {                // The colored bars around the preview window is made up of rectangles
                    // Top bar
                    ImVec2 topBarRectMin(viewAreaElementPosX, viewAreaElementPosY);
                    ImVec2 topBarRectMax(viewAreaElementPosX + info->viewAreaElementSizeX,
                                         viewAreaElementPosY + (info->previewBorderPadding / 2));
                    ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMin, topBarRectMax, ImColor(AR::CRLBlueIsh),
                                                              0.0f,
                                                              0);


                    // extension of top bar in different color
                    ImVec2 topBarRectMinExtended(viewAreaElementPosX, topBarRectMax.y);
                    ImVec2 topBarRectMaxExtended(viewAreaElementPosX + info->viewAreaElementSizeX,
                                                 topBarRectMax.y + 5.0f);

                    ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMinExtended, topBarRectMaxExtended,
                                                              ImColor(AR::CRLGray424Main), 0.0f,
                                                              0);

                    // Left bar

                    ImVec2 leftBarMin(viewAreaElementPosX, topBarRectMaxExtended.y);
                    ImVec2 leftBarMax(viewAreaElementPosX + info->previewBorderPadding,
                                      topBarRectMaxExtended.y + info->viewAreaElementSizeY -
                                      (info->previewBorderPadding * 0.5f));

                    ImGui::GetWindowDrawList()->AddRectFilled(leftBarMin, leftBarMax, ImColor(AR::CRLGray424Main), 0.0f,
                                                              0);

                    // Right bar
                    ImVec2 rightBarMin(
                            viewAreaElementPosX + info->viewAreaElementSizeX - info->previewBorderPadding,
                            topBarRectMaxExtended.y);
                    ImVec2 rightBarMax(viewAreaElementPosX + info->viewAreaElementSizeX,
                                       topBarRectMaxExtended.y + info->viewAreaElementSizeY -
                                       (info->previewBorderPadding * 0.5f));

                    ImGui::GetWindowDrawList()->AddRectFilled(rightBarMin, rightBarMax, ImColor(AR::CRLGray424Main),
                                                              0.0f,
                                                              0);

                    // Bottom bar
                    ImVec2 bottomBarMin(viewAreaElementPosX, topBarRectMaxExtended.y + info->viewAreaElementSizeY -
                                                             (info->previewBorderPadding) - 13.0f);
                    ImVec2 bottomBarMax(viewAreaElementPosX + info->viewAreaElementSizeX,
                                        topBarRectMaxExtended.y + info->viewAreaElementSizeY -
                                        (info->previewBorderPadding * 0.5));

                    ImGui::GetWindowDrawList()->AddRectFilled(bottomBarMin, bottomBarMax, ImColor(AR::CRLGray424Main),
                                                              0.0f,
                                                              0);



                // Max X and Min Y is top right corner
                ImGui::SetCursorScreenPos(ImVec2(topBarRectMax.x - 225.0f, topBarRectMin.y + 5.0f));
                }
                ImGui::Text("Source: ");
                ImGui::SameLine(0.0f, 5.0f);

                ImGui::SetNextItemWidth(150.0f);
                std::string srcLabel = "##Source" + std::to_string(index);

                std::string previewValue;
                auto search = dev.selectedSourceMap.find(index);
                if (search == dev.selectedSourceMap.end())
                    previewValue = "None";
                else
                    previewValue = dev.selectedSourceMap.at(index);

                if (ImGui::BeginCombo(srcLabel.c_str(), previewValue.c_str(),
                                      ImGuiComboFlags_HeightSmall)) {
                    for (int n = 0; n < dev.sources.size(); n++) {

                        const bool is_selected = (dev.selectedSourceIndex == n);
                        if (ImGui::Selectable(dev.sources[n].c_str(), is_selected)) {

                            if (Utils::removeFromVector(&dev.userRequestedSources, dev.selectedSource)) {
                                Log::Logger::getInstance()->info("Removed source '{}' from user requested sources",
                                                                 dev.selectedSource);
                            }

                            dev.selectedSourceIndex = n;
                            dev.selectedSource = dev.sources[dev.selectedSourceIndex];
                            dev.selectedSourceMap[index] = dev.sources[dev.selectedSourceIndex];

                            dev.update = true;

                            if (!Utils::isInVector(dev.userRequestedSources, dev.selectedSource)) {
                                dev.userRequestedSources.emplace_back(dev.selectedSource);
                                Log::Logger::getInstance()->info("Added source '{}' to user requested sources",
                                                                 dev.selectedSource);
                            }
                        }
                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                dev.playbackStatus = AR_PREVIEW_PLAYING;
                ImGui::PopStyleColor();
                ImGui::End();
                index++;
            }
        }
    }

    void createControlArea(AR::GuiObjectHandles *handles, AR::Element &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, AR::CRLCoolGray);
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

                        if (dev.selectedPreviewTab == TAB_2D_PREVIEW) {

                            ImVec2 size = ImVec2(65.0f,
                                                 50.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
                            ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                            ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                            ImVec4 bg_col = AR::CRLCoolGray;         // Match bg color
                            ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                            ImGui::Dummy(ImVec2(40.0f, 80.0f));

                            // Text
                            ImGui::Dummy(ImVec2(40.0f, 0.0));
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextGray);
                            ImGui::Text("1. Choose Layout");
                            ImGui::PopStyleColor();

                            // Image buttons
                            ImGui::Dummy(ImVec2(40.0f, 0.0));
                            ImGui::SameLine();
                            if (ImGui::ImageButton("Single", handles->info->imageButtonTextureDescriptor[6], size, uv0,
                                                   uv1,
                                                   bg_col, tint_col)) {
                                Log::Logger::getInstance()->info("Single Layout pressed.");
                                dev.layout = PREVIEW_LAYOUT_SINGLE;
                            }
                            ImGui::SameLine(0, 20.0f);
                            if (ImGui::ImageButton("Double", handles->info->imageButtonTextureDescriptor[7], size, uv0,
                                                   uv1,
                                                   bg_col, tint_col)) {
                                dev.layout = PREVIEW_LAYOUT_DOUBLE;
                            }
                            ImGui::SameLine(0, 20.0f);
                            if (ImGui::ImageButton("Quad", handles->info->imageButtonTextureDescriptor[8], size, uv0,
                                                   uv1,
                                                   bg_col, tint_col))
                                dev.layout = PREVIEW_LAYOUT_QUAD;

                            ImGui::SameLine(0, 20.0f);
                            if (ImGui::ImageButton("Nine", handles->info->imageButtonTextureDescriptor[9], size, uv0,
                                                   uv1,
                                                   bg_col, tint_col))
                                dev.layout = PREVIEW_LAYOUT_NINE;


                            if (dev.selectedPreviewTab == TAB_2D_PREVIEW && dev.layout != PREVIEW_LAYOUT_NONE) {
                                createDoubleWindowPreview(handles, dev);
                            }


                            ImGui::Dummy(ImVec2(40.0f, 0.0));
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextGray);
                            ImGui::Text("2. Choose Sensor Resolution");
                            ImGui::PopStyleColor();
                            ImGui::Dummy(ImVec2(40.0f, 0.0));
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(200);
                            std::string resLabel = "##Resolution";
                            if (ImGui::BeginCombo(resLabel.c_str(), dev.modes[dev.selectedModeIndex].c_str(),
                                                  ImGuiComboFlags_HeightSmall)) {
                                for (int n = 0; n < dev.modes.size(); n++) {
                                    const bool is_selected = (dev.selectedModeIndex == n);
                                    if (ImGui::Selectable(dev.modes[n].c_str(), is_selected)) {
                                        dev.selectedModeIndex = n;
                                        dev.selectedMode = Utils::stringToCameraResolution(
                                                dev.modes[dev.selectedModeIndex]);
                                        dev.update = true;

                                    }
                                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                                    if (is_selected) {
                                        ImGui::SetItemDefaultFocus();
                                    }
                                }
                                ImGui::EndCombo();
                            }
                        } else if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD) {
                            ImGui::Dummy(ImVec2(40.0f, 40.0));
                            ImGui::Dummy(ImVec2(40.0f, 0.0));
                            ImGui::SameLine();
                            ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextGray);
                            ImGui::Text("1. Choose Sensor Resolution");
                            ImGui::PopStyleColor();
                            ImGui::Dummy(ImVec2(40.0f, 0.0));
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(200);
                            std::string resLabel = "##Resolution";
                            if (ImGui::BeginCombo(resLabel.c_str(), dev.modes[dev.selectedModeIndex].c_str(),
                                                  ImGuiComboFlags_HeightSmall)) {
                                for (int n = 0; n < dev.modes.size(); n++) {
                                    const bool is_selected = (dev.selectedModeIndex == n);
                                    if (ImGui::Selectable(dev.modes[n].c_str(), is_selected)) {
                                        dev.selectedModeIndex = n;
                                        dev.selectedMode = Utils::stringToCameraResolution(
                                                dev.modes[dev.selectedModeIndex]);
                                        dev.update = true;
                                        dev.playbackStatus = AR_PREVIEW_PLAYING;
                                    }
                                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                                    if (is_selected) {
                                        ImGui::SetItemDefaultFocus();
                                    }
                                }
                                ImGui::EndCombo();
                            }

                            dev.selectedSource = "Disparity Left";
                        }


                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Configuration")) {

                        bool active = false;

                        d.parameters.update;
                        ImGui::Checkbox("Auto Exposure", &d.parameters.ep.autoExposure);
                        active |= ImGui::IsItemDeactivated();

                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        if (!d.parameters.ep.autoExposure) {
                            ImGui::SliderInt("Exposure", reinterpret_cast<int *>(&d.parameters.ep.exposure),
                                             10, 30000);
                            active |= ImGui::IsItemDeactivated();

                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        } else {
                            ImGui::SliderInt("Auto exposure max value",
                                             reinterpret_cast<int *>(&d.parameters.ep.autoExposureMax), 10,
                                             35000);
                            active |= ImGui::IsItemDeactivated();

                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            ImGui::SliderInt("Auto exposure decay",
                                             reinterpret_cast<int *>(&d.parameters.ep.autoExposureDecay), 1,
                                             100);
                            active |= ImGui::IsItemDeactivated();

                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            ImGui::SliderFloat("Auto exposure intensity",
                                               &d.parameters.ep.autoExposureTargetIntensity, 0, 1);
                            active |= ImGui::IsItemDeactivated();

                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            ImGui::SliderFloat("Auto exposure threshold", &d.parameters.ep.autoExposureThresh,
                                               0, 1);
                            active |= ImGui::IsItemDeactivated();

                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        }

                        ImGui::Dummy(ImVec2(0.0f, 20.0f));
                        ImGui::Checkbox("Auto White Balance", &d.parameters.wb.autoWhiteBalance);
                        active |= ImGui::IsItemDeactivated();
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        if (!d.parameters.wb.autoWhiteBalance) {
                            ImGui::SliderFloat("White balance red", &d.parameters.wb.whiteBalanceRed, 0, 5);
                            active |= ImGui::IsItemDeactivated();
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            ImGui::SliderFloat("White balance blue", &d.parameters.wb.whiteBalanceBlue, 0,
                                               5);
                            active |= ImGui::IsItemDeactivated();
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        } else {
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            ImGui::SliderFloat("Auto white balance threshold",
                                               &d.parameters.wb.autoWhiteBalanceThresh, 0, 1);
                            active |= ImGui::IsItemDeactivated();
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                            ImGui::SliderFloat("Auto white balance decay",
                                               reinterpret_cast<float *>(&d.parameters.wb.autoWhiteBalanceDecay),
                                               1, 2);
                            active |= ImGui::IsItemDeactivated();
                            ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        }

                        ImGui::SliderFloat("Stereo Post Filter Strength",
                                           &d.parameters.stereoPostFilterStrength, 0, 1);
                        active |= ImGui::IsItemDeactivated();
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        ImGui::SliderFloat("Gamma", &d.parameters.gamma, 1.0f, 2.5f);
                        active |= ImGui::IsItemDeactivated();
                        ImGui::Dummy(ImVec2(1.0f, 2.5f));
                        ImGui::SliderFloat("Gain", &d.parameters.gain, 0, 10);
                        active |= ImGui::IsItemDeactivated();
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));
                        ImGui::SliderFloat("FPS", &d.parameters.fps, 0, 31);
                        active |= ImGui::IsItemDeactivated();
                        ImGui::Dummy(ImVec2(0.0f, 10.0f));

                        ImGui::Dummy(ImVec2(0.0f, handles->info->height - ImGui::GetCursorScreenPos().y));

                        if (active)
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

        // Draw border between control and viewing area
        ImVec2 lineMin(handles->info->sidebarWidth + handles->info->controlAreaWidth - 3.0f, 0.0f);
        ImVec2 lineMax(lineMin.x + 3.0f, handles->info->height);
        ImGui::GetWindowDrawList()->AddRectFilled(lineMin, lineMax, ImColor(AR::CRLGray424Main), 0.0f,
                                                  0);


        ImGui::Dummy(ImVec2(0.0f, handles->info->height - ImGui::GetCursorPosY()));
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
                    stream.selectedStreamingMode = Utils::stringToCameraResolution(
                            stream.modes[stream.selectedModeIndex]);

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
