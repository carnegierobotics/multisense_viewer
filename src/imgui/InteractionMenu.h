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
            ImGui::ShowDemoWindow();
            ImGui::End();
            ImGui::PopStyleColor(); // bg color
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


        ImGui::PopStyleColor(); // bg color
        ImGui::End();

    }


    void buildConfigurationPreview(AR::GuiObjectHandles *handles) {
        //ImGui::ShowDemoWindow();

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
            ImGui::PushStyleColor(ImGuiCol_Button, AR::CRLRedActive);
        else
            ImGui::PushStyleColor(ImGuiCol_Button, AR::CRLRed);

        if (ImGui::Button("2D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = TAB_2D_PREVIEW;
            Log::Logger::getInstance()->info("Profile {}: 2D preview pressed", dev.name.c_str());
        }
        ImGui::PopStyleColor();

        if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD)
            ImGui::PushStyleColor(ImGuiCol_Button, AR::CRLRedActive);
        else
            ImGui::PushStyleColor(ImGuiCol_Button, AR::CRLRed);

        ImGui::SameLine();
        if (ImGui::Button("3D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = TAB_3D_POINT_CLOUD;
            Log::Logger::getInstance()->info("Profile {}: 3D preview pressed", dev.name.c_str());
        }
        ImGui::PopStyleColor();
        ImGui::End();

        ImGui::PopStyleColor(); // Bg color



    }

    void createDoubleWindowPreview(AR::GuiObjectHandles *handles, AR::Element &dev) {
        AR::GuiLayerUpdateInfo *info = handles->info;
        info->viewingAreaWidth = info->width - info->sidebarWidth - info->controlAreaWidth;
        // The top left corner of the ImGui window that encapsulates the quad with the texture playing.
        // Equation is a (to-do)
        info->hoverState = ImGui::IsWindowHoveredByName("ControlArea", ImGuiHoveredFlags_AnyWindow);
        if (!info->hoverState && !handles->input->getButton(GLFW_KEY_LEFT_CONTROL)) {
            handles->accumulatedActiveScroll -= ImGui::GetIO().MouseWheel * 100.0f;
        }
        int cols = 0, rows = 0;
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
                info->previewBorderPadding = 35.0f * (info->width / 1280);


                float offsetX;
                if (dev.layout == PREVIEW_LAYOUT_DOUBLE || dev.layout == PREVIEW_LAYOUT_SINGLE) {
                    offsetX = (info->controlAreaWidth + info->sidebarWidth + ((info->viewingAreaWidth - newWidth) / 2));
                } else
                    offsetX = (info->controlAreaWidth + info->sidebarWidth + 5.0f);

                float viewAreaElementPosX = offsetX + ((float) col * (newWidth + 10.0f));


                ImGui::SetNextWindowSize(ImVec2(info->viewAreaElementSizeX, info->viewAreaElementSizeY),
                                         ImGuiCond_Always);

                dev.row[index] = float(row);
                dev.col[index] = float(col);
                // Calculate window position
                float viewAreaElementPosY =
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


                    /*
                    // extension of top bar in different color
                    ImVec2 topBarRectMinExtended(viewAreaElementPosX, topBarRectMax.y);
                    ImVec2 topBarRectMaxExtended(viewAreaElementPosX + info->viewAreaElementSizeX,
                                                 topBarRectMax.y + 5.0f);

                    ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMinExtended, topBarRectMaxExtended,
                                                              ImColor(AR::CRLGray424Main), 0.0f,
                                                              0);
                    */
                    // Left bar

                    ImVec2 leftBarMin(viewAreaElementPosX, topBarRectMax.y);
                    ImVec2 leftBarMax(viewAreaElementPosX + (info->previewBorderPadding / 2),
                                      topBarRectMax.y + info->viewAreaElementSizeY -
                                      (info->previewBorderPadding * 0.5f));

                    ImGui::GetWindowDrawList()->AddRectFilled(leftBarMin, leftBarMax, ImColor(AR::CRLGray424Main), 0.0f,
                                                              0);

                    // Right bar
                    ImVec2 rightBarMin(
                            viewAreaElementPosX + info->viewAreaElementSizeX - (info->previewBorderPadding / 2),
                            topBarRectMax.y);
                    ImVec2 rightBarMax(viewAreaElementPosX + info->viewAreaElementSizeX,
                                       topBarRectMax.y + info->viewAreaElementSizeY -
                                       (info->previewBorderPadding * 0.5f));

                    ImGui::GetWindowDrawList()->AddRectFilled(rightBarMin, rightBarMax, ImColor(AR::CRLGray424Main),
                                                              0.0f,
                                                              0);

                    // Bottom bar
                    ImVec2 bottomBarMin(viewAreaElementPosX, topBarRectMin.y + info->viewAreaElementSizeY -
                                                             (info->previewBorderPadding / 2.0f));
                    ImVec2 bottomBarMax(viewAreaElementPosX + info->viewAreaElementSizeX,
                                        topBarRectMin.y + info->viewAreaElementSizeY +
                                        (info->previewBorderPadding / 2.0f));

                    ImGui::GetWindowDrawList()->AddRectFilled(bottomBarMin, bottomBarMax, ImColor(AR::CRLGray424Main),
                                                              0.0f,
                                                              0);

                    // Min Y and Min Y is top left corner
                    ImGui::SetCursorScreenPos(ImVec2(topBarRectMin.x + 20.0f, topBarRectMin.y + 5.0f));
                    if (dev.pixelInfoEnable) {
                        // Also only if a window is hovered
                        if (ImGui::IsWindowHoveredByName(std::string("View Area") + std::to_string(index),
                                                         ImGuiHoveredFlags_AnyWindow)) {
                            // Offsset cursor positions.
                            uint32_t x, y, val = dev.pixelInfo.intensity;
                            x = static_cast<uint32_t>( dev.pixelInfo.x - viewAreaElementPosX);
                            y = static_cast<uint32_t>(dev.pixelInfo.y - viewAreaElementPosY);


                            ImGui::Text("(%d, %d) %d", x, y, val);
                        }
                    }


                    // Max X and Min Y is top right corner
                    ImGui::SetCursorScreenPos(ImVec2(topBarRectMax.x - 325.0f, topBarRectMin.y));
                }

                ImGui::Text("Source: ");
                ImGui::SameLine(0.0f, 5.0f);

                ImGui::SetNextItemWidth(250.0f);
                std::string srcLabel = "##Source" + std::to_string(index);

                std::string previewValue;

                ImGui::PushStyleColor(ImGuiCol_PopupBg, AR::CRLBlueIsh);

                if (ImGui::BeginCombo(srcLabel.c_str(), dev.sources[dev.selectedSourceIndexMap[index]].c_str(),
                                      ImGuiComboFlags_HeightLarge)) {
                    for (int n = 0; n < dev.sources.size(); n++) {

                        const bool is_selected = (dev.selectedSourceIndex == n);
                        if (ImGui::Selectable(dev.sources[n].c_str(), is_selected)) {

                            if (dev.selectedSourceMap.contains(index)) {

                                bool inUse = false;
                                for (const auto &source: dev.selectedSourceMap) {
                                    if (dev.selectedSourceMap[index] == source.second && index != source.first)
                                        inUse = true;
                                }


                                if (!inUse &&
                                    Utils::removeFromVector(&dev.userRequestedSources, dev.selectedSourceMap[index])) {
                                    Log::Logger::getInstance()->info("Removed source '{}' from user requested sources",
                                                                     dev.selectedSourceMap[index]);

                                }
                            }

                            dev.selectedSourceIndex = n;
                            dev.selectedSourceIndexMap[index] = dev.selectedSourceIndex;
                            dev.selectedSourceMap[index] = dev.sources[dev.selectedSourceIndex];


                            if (!Utils::isInVector(dev.userRequestedSources, dev.selectedSourceMap[index])) {
                                dev.userRequestedSources.emplace_back(dev.selectedSourceMap[index]);
                                Log::Logger::getInstance()->info("Added source '{}' to user requested sources",
                                                                 dev.selectedSourceMap[index]);
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
                ImGui::PopStyleColor(); // PopupBg
                ImGui::End();
                index++;
            }
        }
    }

    void drawVideoPreviewGuiOverlay(AR::GuiObjectHandles *handles, AR::Element &dev, bool withStreamControls) {

        if (dev.selectedPreviewTab == TAB_2D_PREVIEW ) {

            if (withStreamControls) {
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

                        }
                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::Dummy(ImVec2(40.0f, 10.0));
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImGui::Checkbox("Display cursor info", &dev.pixelInfoEnable);
            }

            if (dev.selectedPreviewTab == TAB_2D_PREVIEW && dev.layout != PREVIEW_LAYOUT_NONE) {
                createDoubleWindowPreview(handles, dev);
            }

            // Remove these two sources if we switch between 3D to 2D and we are not using the sources in 2D
            std::vector<std::string> pointCloudSources({"Disparity Left", "Luma Rectified Left"});
            for (const auto &source: pointCloudSources) {
                bool inUse = false;
                for (int index = 0; index < AR_PREVIEW_TOTAL_MODES; ++index) {
                    if (!dev.selectedSourceMap.contains(index))
                        continue;
                    if (dev.selectedSourceMap[index] == source)
                        inUse = true;
                }

                if (!inUse && Utils::isInVector(dev.userRequestedSources, source)) {
                    Utils::removeFromVector(&dev.userRequestedSources, source);
                }
            }


        } else if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD && withStreamControls) {
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
                        dev.playbackStatus = AR_PREVIEW_PLAYING;
                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            dev.selectedSourceMap[AR_PREVIEW_POINT_CLOUD] = "Disparity Left";

            if (!Utils::isInVector(dev.userRequestedSources, "Disparity Left")) {
                dev.userRequestedSources.emplace_back("Disparity Left");
            }
            if (!Utils::isInVector(dev.userRequestedSources, "Luma Rectified Left")) {
                dev.userRequestedSources.emplace_back("Luma Rectified Left");
            }
        }


    }

    void createControlArea(AR::GuiObjectHandles *handles, AR::Element &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 15.0f);

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

                        drawVideoPreviewGuiOverlay(handles, dev, true);
                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);

                    if (ImGui::BeginTabItem("Configuration")) {

                        drawVideoPreviewGuiOverlay(handles, dev, false);
                        buildConfigurationTab(handles, dev);
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
        ImGui::PopStyleVar(3);
    }

    void buildConfigurationTab(AR::GuiObjectHandles *handles, AR::Element &d) {
        bool active = false;
        ImGui::ShowDemoWindow();
        // Exposure Tab
        {
            float textSpacing = 90.0f;
            ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextGray);

            ImGui::PushFont(handles->info->font18);
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::Dummy(ImVec2(10.0f, 0.0f));
            ImGui::SameLine();
            ImGui::Text("Exposure");
            ImGui::PopFont();

            ImGui::Dummy(ImVec2(0.0f, 15.0f));
            ImGui::Dummy(ImVec2(25.0f, 0.0f));
            ImGui::SameLine();
            std::string txt = "Enable Auto:";
            ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
            ImGui::Text("%s", txt.c_str());
            ImGui::SameLine(0, textSpacing - txtSize.x);
            ImGui::Checkbox("##Enable Auto Exposure", &d.parameters.ep.autoExposure);
            d.parameters.ep.update = ImGui::IsItemDeactivatedAfterEdit();
            // Draw Manual eposure controls or auto exposure control
            if (!d.parameters.ep.autoExposure) {
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Value:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderInt("##Exposure Value: ", reinterpret_cast<int *>(&d.parameters.ep.exposure),
                                 10, 30000);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            } else {
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Max Value:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderInt("##Max",
                                 reinterpret_cast<int *>(&d.parameters.ep.autoExposureMax), 10,
                                 35000);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Decay Rate:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderInt("##Decay",
                                 reinterpret_cast<int *>(&d.parameters.ep.autoExposureDecay), 0, 20);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Intensity:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##TargetIntensity",
                                   &d.parameters.ep.autoExposureTargetIntensity, 0, 1);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Threshold:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##Threshold", &d.parameters.ep.autoExposureThresh,
                                   0, 1);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }

            // White Balance
            {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("White Balance");
                ImGui::PopFont();

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                std::string txt = "Enable Auto:";
                ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::Checkbox("##EnableAutoWhiteBalance", &d.parameters.wb.autoWhiteBalance);
                d.parameters.wb.update = ImGui::IsItemDeactivatedAfterEdit();
                ImGui::Dummy(ImVec2(0.0f, 5.0f));

                if (!d.parameters.wb.autoWhiteBalance){
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Red Balance:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                    ImGui::SliderFloat("##WBRed",
                                     &d.parameters.wb.whiteBalanceRed, 0.25f,
                                     4.0f);
                    d.parameters.wb.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Blue Balance:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                    ImGui::SliderFloat("##WBBlue",
                                       &d.parameters.wb.whiteBalanceBlue, 0.25f,
                                       4.0f);
                    d.parameters.wb.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();
                } else {
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Threshold:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                    ImGui::SliderFloat("##WBTreshold",
                                       &d.parameters.wb.autoWhiteBalanceThresh, 0.0,
                                       1.0f);
                    d.parameters.wb.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Decay Rate:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                    ImGui::SliderInt("##DecayRateWB",
                                     reinterpret_cast<int *>(&d.parameters.wb.autoWhiteBalanceDecay), 0,
                                     20);
                    d.parameters.wb.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();
                }

            }

            // Lighting controls
            {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("Lighting - Not implemented");
                ImGui::PopFont();

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                std::string txt = "Enable Lights:";
                ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::Checkbox("##Enable Lights", &d.parameters.light.enable);
                d.parameters.light.update = ImGui::IsItemDeactivatedAfterEdit();
                ImGui::Dummy(ImVec2(0.0f, 5.0f));

                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Duty Cycle:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##Duty_Cycle",
                                 &d.parameters.light.dutyCycle, 0,
                                 100);
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Light Index:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderInt("##LightSelection",
                                 reinterpret_cast<int *>(&d.parameters.light.selection), 1,
                                 5);
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();

                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Light Pulses:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderInt("##Pulses",
                                 reinterpret_cast<int *>(&d.parameters.light.numLightPulses), 10,
                                 35000);
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Startup Time:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderInt("##Startup Time",
                                 reinterpret_cast<int *>(&d.parameters.light.startupTime), 1,
                                 100);
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }
            // Additional Params
            {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("Additional Params");
                ImGui::PopFont();

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "HDR";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::Checkbox("##HDREnable", &d.parameters.hdrEnabled);
                d.parameters.update = ImGui::IsItemDeactivatedAfterEdit();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Framerate:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##Framerate",
                                 &d.parameters.fps, 1,
                                 30);
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Gain:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##Gain",
                                 &d.parameters.gain, 0,
                                 3);
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Gamma:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##Gamma",
                                   &d.parameters.gamma, 0,
                                   3);
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Stereo Filter:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, AR::CRLTextWhite);
                ImGui::SliderFloat("##Stereo",
                                 &d.parameters.stereoPostFilterStrength, 0,
                                 1);
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }

            ImGui::PopStyleColor();
        }
    }

};

#endif //MULTISENSE_INTERACTIONMENU_H
