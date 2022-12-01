//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include <GLFW/glfw3.h>
#include <ImGuiFileDialog/ImGuiFileDialog.h>


#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/Layer.h"

#ifdef WIN32

#else

#include <unistd.h>

#endif

class InteractionMenu : public VkRender::Layer {
private:
    bool page[PAGE_TOTAL_PAGES] = {false, false, true};
    bool drawActionPage = false;
    ImGuiFileDialog chooseIntrinsicsDialog;
    ImGuiFileDialog chooseExtrinsicsDialog;
    ImGuiFileDialog saveCalibrationDialog;
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> showSavedTimer;
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> showSetTimer;

// Create global object for convenience in other functions
public:
    void onFinishedRender() override {

    }

    void onDetach() override {

    }

    void onAttach() override {

    }

    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        bool allUnavailable = true;
        for (auto &d: handles->devices) {
            if (d.state == AR_STATE_ACTIVE)
                allUnavailable = false;
        }

        if (allUnavailable) {
            // Use background color again
            handles->clearColor[0] = VkRender::CRLCoolGray.x;
            handles->clearColor[1] = VkRender::CRLCoolGray.y;
            handles->clearColor[2] = VkRender::CRLCoolGray.z;
            handles->clearColor[3] = VkRender::CRLCoolGray.w;

            return;
        }

        // Check if stream was interrupted by a disconnect event and reset pages events across all devices
        for (auto &d: handles->devices) {
            if (d.state == AR_STATE_RESET) {
                std::fill_n(page, PAGE_TOTAL_PAGES, false);
                drawActionPage = false;
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

            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLCoolGray);
            ImGui::Begin("InteractionMenu", &pOpen, window_flags);

            int imageButtonHeight = 100;
            const char *labels[3] = {"Preview Device \n!(Not implemented)", "Device Information \n!(Not implemented)",
                                     "Configure Device"};
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            //ImGui::ShowDemoWindow();

            for (int i = 0; i < PAGE_TOTAL_PAGES; i++) {
                float imageSpacingX = 200.0f;

                // m_Width of menu buttons layout. Needed to draw on center of screen.
                float xOffset = ((handles->info->width - handles->info->sidebarWidth) / 2) -
                                (((float) imageSpacingX * (float) PAGE_CONFIGURE_DEVICE) +
                                 ((float) PAGE_TOTAL_PAGES * 100.0f) / 2) +
                                handles->info->sidebarWidth +
                                ((float) i * imageSpacingX);

                ImGui::SetCursorPos(ImVec2(xOffset,
                                           (handles->info->height / 2) - ((float) imageButtonHeight / 2)));
                float posY = ImGui::GetCursorScreenPos().y;

                ImVec2 size = ImVec2(100.0f,
                                     100.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = VkRender::CRLCoolGray;         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                ImGui::PushID(i);

                ImGui::ImageButton(labels[i], handles->info->imageButtonTextureDescriptor[i], size, uv0, uv1,
                                   bg_col, tint_col);
                ImGui::PopID();
                ImGui::SetItemAllowOverlap();

                ImGui::PushFont(handles->info->font18);
                ImGui::SetCursorPos(ImVec2(xOffset + ((100.0f - ImGui::CalcTextSize(labels[i]).x) / 2),
                                           (handles->info->height / 2) + ((float) imageButtonHeight / 2) + 8));
                float posX = ImGui::GetCursorScreenPos().x;
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Text("%s", labels[i]);
                ImGui::PopStyleColor();


                // Reset cursorpos to first pos
                ImVec2 btnSize = ImVec2(ImGui::CalcTextSize(labels[i]).x,
                                        imageButtonHeight + ImGui::CalcTextSize(labels[i]).y + 10.0f);
                ImGui::PopFont();

                ImVec2 pos(posX, posY);
                ImVec2 posMax = btnSize;
                posMax.x += pos.x + 5.0f;
                posMax.y += pos.y;

                pos.x -= 5.0f;

                ImGui::SetCursorScreenPos(pos);
                bool hovered = false;
                bool held = false;
                if (ImGui::HoveredInvisibleButton(labels[i], &hovered, &held, btnSize, 0))
                    page[i] = true;

                ImGui::GetWindowDrawList()->AddRectFilled(pos, posMax, ImGui::GetColorU32(
                        hovered && held ? VkRender::CRLBlueIshTransparent2 :
                        hovered ? VkRender::CRLBlueIshTransparent : VkRender::CRLCoolGrayTransparent), 5.0f);

                ImGui::SameLine();
            }

            ImGui::PopStyleVar();
            ImGui::NewLine();
            ImGui::End();
            ImGui::PopStyleColor(); // bg color
        }
    }

private:

    void buildDeviceInformation(VkRender::GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLCoolGray);
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_DEVICE_INFORMATION] = false;
            drawActionPage = true;
        }

        ImGui::NewLine();
        ImGui::PopStyleColor(); // bg color
        ImGui::End();
    }

    void buildPreview(VkRender::GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLCoolGray);
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[PAGE_PREVIEW_DEVICES] = false;
            drawActionPage = true;
        }


        ImGui::PopStyleColor(); // bg color
        ImGui::End();

    }


    void buildConfigurationPreview(VkRender::GuiObjectHandles *handles) {
        for (auto &dev: handles->devices) {
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

    void createViewingArea(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {

        /** CREATE TOP NAVIGATION BAR **/
        bool pOpen = true;
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::SetNextWindowPos(
                ImVec2(handles->info->sidebarWidth + handles->info->controlAreaWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLCoolGray);

        handles->info->viewingAreaWidth =
                handles->info->width - handles->info->sidebarWidth - handles->info->controlAreaWidth;

        ImGui::SetNextWindowSize(ImVec2(handles->info->viewingAreaWidth, handles->info->viewingAreaHeight),
                                 ImGuiCond_Always);
        ImGui::Begin("ViewingArea", &pOpen, window_flags);

        ImVec2 backButtonPos = ImGui::GetCursorScreenPos();

        ImGui::Dummy(ImVec2((handles->info->viewingAreaWidth / 2) - 30.0f, 0.0f));
        ImGui::SameLine();

        ImGui::PushFont(handles->info->font18);
        ImGui::Text("Viewing");
        ImGui::PopFont();

        backButtonPos.x += handles->info->viewingAreaWidth - 100.0f;
        ImGui::SetCursorScreenPos(backButtonPos);
        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);
        if (ImGui::Button("Back", ImVec2(100.0f, 20.0f))) {
            page[PAGE_CONFIGURE_DEVICE] = false;
            drawActionPage = true;
        }
        ImGui::PopStyleColor();

        ImGui::Dummy(ImVec2((handles->info->viewingAreaWidth / 2) - 80.0f, 0.0f));
        ImGui::SameLine();

        if (dev.selectedPreviewTab == TAB_2D_PREVIEW)
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRedActive);
        else
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);

        if (ImGui::Button("2D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = TAB_2D_PREVIEW;
            Log::Logger::getInstance()->info("Profile {}: 2D preview pressed", dev.name.c_str());
            handles->clearColor[0] = VkRender::CRLCoolGray.x;
            handles->clearColor[1] = VkRender::CRLCoolGray.y;
            handles->clearColor[2] = VkRender::CRLCoolGray.z;
            handles->clearColor[3] = VkRender::CRLCoolGray.w;
        }
        ImGui::PopStyleColor();

        if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD)
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRedActive);
        else
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);

        ImGui::SameLine();
        if (dev.isRemoteHead) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::TextColorGray);
        }
        if (ImGui::Button("3D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = TAB_3D_POINT_CLOUD;
            Log::Logger::getInstance()->info("Profile {}: 3D preview pressed", dev.name.c_str());
            handles->clearColor[0] = VkRender::CRL3DBackground.x;
            handles->clearColor[1] = VkRender::CRL3DBackground.y;
            handles->clearColor[2] = VkRender::CRL3DBackground.z;
            handles->clearColor[3] = VkRender::CRL3DBackground.w;
        }
        if (dev.isRemoteHead) {
            ImGui::PopItemFlag();
            ImGui::PopStyleColor();
        }
        ImGui::PopStyleColor();
        ImGui::End();

        ImGui::PopStyleColor(); // Bg color



    }

    void createWindowPreviews(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        handles->info->viewingAreaWidth =
                handles->info->width - handles->info->sidebarWidth - handles->info->controlAreaWidth;
        // The top left corner of the ImGui window that encapsulates the quad with the texture playing.
        // Equation is a (to-do)
        handles->info->hoverState = ImGui::IsWindowHoveredByName("ControlArea", ImGuiHoveredFlags_AnyWindow);
        if (!handles->info->hoverState && dev.layout == PREVIEW_LAYOUT_DOUBLE) {
            handles->accumulatedActiveScroll -= ImGui::GetIO().MouseWheel * 100.0f;

            if (handles->accumulatedActiveScroll > handles->maxScroll) {
                handles->accumulatedActiveScroll = handles->maxScroll - 1.0f;
            }
            if (handles->accumulatedActiveScroll < handles->minScroll) {
                handles->accumulatedActiveScroll = handles->minScroll + 1.0f;
            }
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

                float newWidth = (handles->info->width - (640 + (5.0f + (5.0f * (float) cols)))) / (float) cols;
                float newHeight = newWidth * (10.0f / 16.0f); // aspect ratio 16:10 of camera images
                // Calculate window size
                handles->info->viewAreaElementSizeX = newWidth;
                handles->info->viewAreaElementSizeY = newHeight;
                handles->info->previewBorderPadding = 35.0f * (handles->info->width / 1280);


                float offsetX;
                if (dev.layout == PREVIEW_LAYOUT_DOUBLE || dev.layout == PREVIEW_LAYOUT_SINGLE) {
                    offsetX = (handles->info->controlAreaWidth + handles->info->sidebarWidth +
                               ((handles->info->viewingAreaWidth - newWidth) / 2));
                } else
                    offsetX = (handles->info->controlAreaWidth + handles->info->sidebarWidth + 5.0f);

                float viewAreaElementPosX = offsetX + ((float) col * (newWidth + 10.0f));


                ImGui::SetNextWindowSize(
                        ImVec2(handles->info->viewAreaElementSizeX, handles->info->viewAreaElementSizeY),
                        ImGuiCond_Always);

                dev.row[index] = float(row);
                dev.col[index] = float(col);
                // Calculate window m_Position
                float viewAreaElementPosY =
                        handles->info->tabAreaHeight + ((float) row * (handles->info->viewAreaElementSizeY + 10.0f));

                if (dev.layout == PREVIEW_LAYOUT_DOUBLE) {
                    viewAreaElementPosY = viewAreaElementPosY + handles->accumulatedActiveScroll;
                }
                dev.win.at(index).xPixelStartPos = viewAreaElementPosX;
                dev.win.at(index).yPixelStartPos = viewAreaElementPosY;
                ImGui::SetNextWindowPos(ImVec2(viewAreaElementPosX, viewAreaElementPosY),
                                        ImGuiCond_Always);

                static bool open = true;
                ImGuiWindowFlags window_flags =
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus;

                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                ImGui::Begin((std::string("View Area") + std::to_string(index)).c_str(), &open, window_flags);

                // The colored bars around the preview window is made up of rectangles
                // Top bar
                ImVec2 topBarRectMin(viewAreaElementPosX, viewAreaElementPosY);
                ImVec2 topBarRectMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                                     viewAreaElementPosY + (handles->info->previewBorderPadding / 2));
                ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMin, topBarRectMax, ImColor(VkRender::CRLBlueIsh),
                                                          0.0f,
                                                          0);

                // left bar

                ImVec2 leftBarMin(viewAreaElementPosX, topBarRectMax.y);
                ImVec2 leftBarMax(viewAreaElementPosX + (handles->info->previewBorderPadding / 2),
                                  topBarRectMax.y + handles->info->viewAreaElementSizeY -
                                  (handles->info->previewBorderPadding * 0.5f));

                ImGui::GetWindowDrawList()->AddRectFilled(leftBarMin, leftBarMax, ImColor(VkRender::CRLGray424Main),
                                                          0.0f,
                                                          0);

                // Right bar
                ImVec2 rightBarMin(
                        viewAreaElementPosX + handles->info->viewAreaElementSizeX -
                        (handles->info->previewBorderPadding / 2),
                        topBarRectMax.y);
                ImVec2 rightBarMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                                   topBarRectMax.y + handles->info->viewAreaElementSizeY -
                                   (handles->info->previewBorderPadding * 0.5f));

                ImGui::GetWindowDrawList()->AddRectFilled(rightBarMin, rightBarMax, ImColor(VkRender::CRLGray424Main),
                                                          0.0f,
                                                          0);

                // Bottom bar
                ImVec2 bottomBarMin(viewAreaElementPosX, topBarRectMin.y + handles->info->viewAreaElementSizeY -
                                                         (handles->info->previewBorderPadding / 2.0f));
                ImVec2 bottomBarMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                                    topBarRectMin.y + handles->info->viewAreaElementSizeY +
                                    (handles->info->previewBorderPadding / 2.0f));

                ImGui::GetWindowDrawList()->AddRectFilled(bottomBarMin, bottomBarMax,
                                                          ImColor(VkRender::CRLGray424Main),
                                                          0.0f,
                                                          0);

                // Min Y and Min Y is top left corner

                ImGui::SetCursorScreenPos(ImVec2(topBarRectMin.x + 20.0f, topBarRectMin.y + 5.0f));
                if (dev.pixelInfoEnable) {
                    // Also only if a window is hovered
                    if (ImGui::IsWindowHoveredByName(std::string("View Area") + std::to_string(index),
                                                     ImGuiHoveredFlags_AnyWindow)) {
                        // Offsset cursor positions.
                        switch (Utils::CRLSourceToTextureType(dev.win.at(index).selectedSource)) {
                            case AR_POINT_CLOUD:
                                break;
                            case AR_GRAYSCALE_IMAGE:
                                ImGui::Text("(%d, %d) %d", dev.pixelInfo.x, dev.pixelInfo.y, dev.pixelInfo.intensity);
                                break;
                            case AR_COLOR_IMAGE:
                                break;
                            case AR_COLOR_IMAGE_YUV420:
                                break;
                            case AR_YUV_PLANAR_FRAME:
                                break;
                            case AR_CAMERA_IMAGE_NONE:
                                break;
                            case AR_DISPARITY_IMAGE:
                                ImGui::Text("(%d, %d) %.3f", dev.pixelInfo.x, dev.pixelInfo.y, dev.pixelInfo.depth);
                                break;
                        }

                    }
                }


                // Max X and Min Y is top right corner
                ImGui::SetCursorScreenPos(ImVec2(topBarRectMax.x - 235.0f, topBarRectMin.y));
                ImGui::SetNextItemWidth(80.0f);
                auto &window = dev.win[index];

                if (dev.isRemoteHead) {
                    ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::CRLBlueIsh);
                    std::string label = window.availableRemoteHeads[Utils::getIndexOf(window.availableRemoteHeads,
                                                                                      std::to_string(
                                                                                              window.selectedRemoteHeadIndex))];
                    std::string comboLabel = "##RemoteHeadSelection" + std::to_string(index);
                    if (ImGui::BeginCombo(comboLabel.c_str(), label.c_str(),
                                          ImGuiComboFlags_HeightLarge)) {
                        for (size_t n = 0; n < window.availableRemoteHeads.size(); n++) {
                            const bool is_selected = (window.selectedRemoteHeadIndex ==
                                                      (crl::multisense::RemoteHeadChannel) n);
                            if (ImGui::Selectable(window.availableRemoteHeads[n].c_str(), is_selected)) {


                                // Disable the previously enabled source if not in use and update the selected source tab
                                if (window.selectedSource != "Source") {
                                    bool inUse = false;
                                    std::string sourceInUse;
                                    for (const auto &win: dev.win) {
                                        if (win.second.selectedSource == window.selectedSource &&
                                            (int) win.first != index &&
                                            win.second.selectedRemoteHeadIndex == window.selectedRemoteHeadIndex) {
                                            inUse = true;
                                            sourceInUse = win.second.selectedSource;
                                        }
                                    }
                                    // If it's not in use we can disable it
                                    if (!inUse && Utils::removeFromVector(
                                            &dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                            window.selectedSource)) {
                                        Log::Logger::getInstance()->info(
                                                "Removed source '{}' from user requested sources",
                                                window.selectedSource);
                                        window.selectedSource = window.availableSources[0]; // 0th index is always "Source"
                                        window.selectedSourceIndex = 0;
                                    }
                                    // If its in use, but we don't want to disable it. Just reset the source name to "Source"
                                    if (inUse && !sourceInUse.empty()) {
                                        window.selectedSource = window.availableSources[0]; // 0th index is always "Source"
                                        window.selectedSourceIndex = 0;
                                    }
                                }

                                window.selectedRemoteHeadIndex = (crl::multisense::RemoteHeadChannel) std::stoi(
                                        window.availableRemoteHeads[n]);
                                Log::Logger::getInstance()->info("Selected Remote head number '{}' for preview {}",
                                                                 window.selectedRemoteHeadIndex, index);
                            }

                            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if (is_selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::PopStyleColor();
                }

                // Set the avaiable sources according to the selected remote head We have "Select Head" in the list as well
                window.availableSources = dev.channelInfo[window.selectedRemoteHeadIndex].availableSources;
                ImGui::SetCursorScreenPos(ImVec2(topBarRectMax.x - 150.0f, topBarRectMin.y));
                ImGui::SetNextItemWidth(150.0f);
                std::string srcLabel = "##Source" + std::to_string(index);
                std::string previewValue;
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::CRLBlueIsh);
                std::vector<std::string> colorSources = {"Color Aux", "Color Rectified Aux"};
                std::vector<std::string> auxLumaSources = {"Luma Aux", "Luma Rectified Aux"};
                if (ImGui::BeginCombo(srcLabel.c_str(), window.selectedSource.c_str(),
                                      ImGuiComboFlags_HeightLarge)) {
                    for (size_t n = 0; n < window.availableSources.size(); n++) {
                        const bool is_selected = (window.selectedSourceIndex == n);
                        if (ImGui::Selectable(window.availableSources[n].c_str(), is_selected)) {

                            if (window.selectedSource != "Source") {
                                bool inUse = false;
                                for (const auto &win: dev.win) {
                                    if (win.second.selectedSource == window.selectedSource &&
                                        (int) win.first != index &&
                                        win.second.selectedRemoteHeadIndex == window.selectedRemoteHeadIndex) {
                                        inUse = true;
                                    }

                                    // If a color source is active in another window and our selected source is a aux *luma source then do nothing
                                    if (win.first != index &&
                                        Utils::isInVector(colorSources, win.second.selectedSource) &&
                                        Utils::isInVector(auxLumaSources, window.selectedSource)) {
                                        inUse = true;
                                    }
                                    // It's in use if we have color aux running but we are disabling luma aux

                                }
                                if (!inUse && Utils::removeFromVector(
                                        &dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                        window.selectedSource)) {
                                    Log::Logger::getInstance()->info("Removed source '{}' from user requested sources",
                                                                     window.selectedSource);
                                }
                            }


                            window.selectedSourceIndex = static_cast<uint32_t>(n);
                            window.selectedSource = window.availableSources[window.selectedSourceIndex];
                            Log::Logger::getInstance()->info("Selected source '{}' for preview {},",
                                                             window.selectedSource, index);

                            if (!Utils::isInVector(dev.channelInfo[window.selectedRemoteHeadIndex].enabledStreams,
                                                   window.selectedSource) && window.selectedSource != "Source") {
                                dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams.emplace_back(
                                        window.selectedSource);
                                Log::Logger::getInstance()->info(
                                        "Added source '{}' from head {} to user requested sources",
                                        window.selectedSource, window.selectedRemoteHeadIndex);
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
                /** Color rest of area in the background color exluding previews**/
                ImGui::End();
                index++;
            }
        }
    }

    void drawVideoPreviewGuiOverlay(VkRender::GuiObjectHandles *handles, VkRender::Device &dev,
                                    bool withStreamControls) {

        if (dev.selectedPreviewTab == TAB_2D_PREVIEW) {

            if (withStreamControls) {
                ImVec2 size = ImVec2(65.0f,
                                     50.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = VkRender::CRLCoolGray;         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
                ImGui::Dummy(ImVec2(40.0f, 40.0f));

                // Text
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImGui::PushFont(handles->info->font18);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Text("1. Choose Layout");
                ImGui::PopStyleColor();

                // Image buttons
                ImGui::Dummy(ImVec2(00.0f, 5.0));
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
                /*
                ImGui::SameLine(0, 20.0f);

                if (ImGui::ImageButton("Nine", handles->info->imageButtonTextureDescriptor[9], size, uv0,
                                       uv1,
                                       bg_col, tint_col))
                    dev.layout = PREVIEW_LAYOUT_NINE;

                 */
                ImGui::Dummy(ImVec2(00.0f, 10.0));
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Text("2. Choose Sensor Resolution");
                ImGui::PopStyleColor();
                ImGui::PopFont();
                ImGui::Dummy(ImVec2(00.0f, 7.0));

                for (size_t i = 0; i < dev.channelInfo.size(); ++i) {
                    if (dev.channelInfo[i].state != AR_STATE_ACTIVE)
                        continue;

                    ImGui::Dummy(ImVec2(40.0f, 0.0));
                    ImGui::SameLine();
                    if (dev.isRemoteHead) {
                        std::string descriptionText = "Remote head " + std::to_string(i) + ":";
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                        ImGui::Text("%s", descriptionText.c_str());
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                    }

                    // Resolution selection box
                    ImGui::SetNextItemWidth(250);
                    std::string resLabel = "##Resolution" + std::to_string(i);
                    auto &chInfo = dev.channelInfo[i];
                    if (chInfo.state != AR_STATE_ACTIVE)
                        continue;
                    if (ImGui::BeginCombo(resLabel.c_str(),
                                          Utils::cameraResolutionToString(chInfo.selectedMode).c_str(),
                                          ImGuiComboFlags_HeightSmall)) {
                        for (size_t n = 0; n < chInfo.modes.size(); n++) {
                            const bool is_selected = (chInfo.selectedModeIndex == n);
                            if (ImGui::Selectable(chInfo.modes[n].c_str(), is_selected)) {
                                chInfo.selectedModeIndex = static_cast<uint32_t>(n);
                                chInfo.selectedMode = Utils::stringToCameraResolution(
                                        chInfo.modes[chInfo.selectedModeIndex]);
                                chInfo.updateResolutionMode = true;

                            }
                            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if (is_selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }
                }
                // Draw Recording options
                {
                    ImGui::Dummy(ImVec2(0.0f, 50.0f));
                    ImVec2 posMin = ImGui::GetCursorScreenPos();
                    ImVec2 posMax = posMin;
                    posMax.x += handles->info->controlAreaWidth;
                    posMax.y += 2.0f;
                    ImGui::GetWindowDrawList()->AddRectFilled(posMin, posMax, ImColor(VkRender::CRLGray421));

                    ImGui::Dummy(ImVec2(0.0f, 30.0f));
                    ImGui::Dummy(ImVec2(40.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushFont(handles->info->font18);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                    ImGui::Text("Recording");
                    ImGui::PopFont();
                    ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                    ImGui::HelpMarker(
                            " \n Saves the frames shown in the viewing are to the right to files.  \n Each type of stream is saved in separate folders \n Depending on hardware, active streams, and if you chose \n a compressed method (png)    \n you may not be able to save all frames \n ");
                    // if start then show gif spinner
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(40.0f, 0.0f));
                    ImGui::SameLine();
                    ImVec2 btnSize(120.0f, 30.0f);
                    std::string btnText = dev.isRecording ? "Stop" : "Start";
                    if (ImGui::Button(btnText.c_str(), btnSize) && dev.outputSaveFolder != "/Path/To/Folder/") {
                        dev.isRecording = !dev.isRecording;
                    }
                    ImGui::SameLine();

                    static std::vector<std::string> saveFormat = {"tiff", "png"};
                    static size_t selector = 0;

                    ImGui::SetNextItemWidth(
                            handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);
                    std::string previewLabel = "Uncompressed (Tiff)";
                    if (ImGui::BeginCombo("##Compression", saveFormat[selector].c_str(), ImGuiComboFlags_HeightSmall)) {
                        for (size_t n = 0; n < saveFormat.size(); n++) {
                            const bool is_selected = (selector == n);
                            if (ImGui::Selectable(saveFormat[n].c_str(), is_selected)) {
                                selector = n;
                                dev.saveImageCompressionMethod = saveFormat[selector];
                            }
                            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if (is_selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::Dummy(ImVec2(40.0f, 0.0f));
                    ImGui::SameLine();
                    // open Dialog Simple
                    if (dev.isRecording) {
                        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::TextColorGray);
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::TextColorGray);

                    }
                    {
                        if (ImGui::Button("Choose Location", btnSize)) {
                            ImGuiFileDialog::Instance()->OpenDialog("ChooseDirDlgKey", "Choose a Directory", nullptr,
                                                                    ".");
                        }
                    }

                    ImGui::SameLine();
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 9.0f));
                    ImGui::SetNextItemWidth(
                            handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);

                    ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::CRLTextWhiteDisabled);
#ifdef WIN32
                    std::string hint = "C:\\Path\\To\\Dir";
#else
                    std::string hint = "/Path/To/Dir";
#endif
                    ImGui::CustomInputTextWithHint("##SaveFolderLocation", hint.c_str(), &dev.outputSaveFolder,
                                                   ImGuiInputTextFlags_AutoSelectAll);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleVar();

                    if (dev.isRecording) {
                        ImGui::PopStyleColor(2);
                        ImGui::PopItemFlag();
                    }

                    // display
                    //ImGui::SetNextWindowPos(ImGui::GetCursorScreenPos());
                    //ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
                    ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLDarkGray425);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                    if (ImGuiFileDialog::Instance()->Display("ChooseDirDlgKey", 0, ImVec2(600.0f, 400.0f),
                                                             ImVec2(1200.0f, 1000.0f))) {
                        // action if OK
                        if (ImGuiFileDialog::Instance()->IsOk()) {
                            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                            dev.outputSaveFolder = filePathName;
                            // action
                        }

                        // close
                        ImGuiFileDialog::Instance()->Close();
                    }
                    ImGui::PopStyleColor();
                    ImGui::PopStyleVar();
                }

            }

            if (dev.selectedPreviewTab == TAB_2D_PREVIEW && dev.layout != PREVIEW_LAYOUT_NONE) {
                createWindowPreviews(handles, dev);
            }

            // Some Point cloud enable/disable sources logic
            // Remove these two sources if we switch between 3D to 2D and we are not using the sources in 2D
            std::vector<std::string> pointCloudSources({"Disparity Left", "Luma Rectified Left"});
            // Disable IMU as well in 2D
            auto &chInfo = dev.channelInfo.front();
            if (dev.useImuData)
                pointCloudSources.emplace_back("IMU");
            for (const auto &source: pointCloudSources) {
                bool inUse = false;
                // Loop over all previews and check their source.
                // If it matches either point cloud source then it means it is in use
                for (const auto &preview: dev.win) {
                    if (preview.second.selectedSource == source && preview.first != AR_PREVIEW_POINT_CLOUD)
                        inUse = true;
                }
                if (!inUse && Utils::isInVector(chInfo.requestedStreams, source)) {
                    Utils::removeFromVector(&chInfo.requestedStreams, source);
                    Log::Logger::getInstance()->info(
                            "Removed {} from user requested sources because it is not in use anymore", source);
                }
            }


        } else if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD && withStreamControls) {
            ImGui::Dummy(ImVec2(40.0f, 40.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
            ImGui::Text("1. Choose Sensor Resolution");
            ImGui::PopStyleColor();
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::SetNextItemWidth(200);
            std::string resLabel = "##Resolution";
            auto &chInfo = dev.channelInfo.front();
            dev.playbackStatus = AR_PREVIEW_PLAYING;
            if (ImGui::BeginCombo(resLabel.c_str(), Utils::cameraResolutionToString(chInfo.selectedMode).c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (size_t n = 0; n < chInfo.modes.size(); n++) {
                    const bool is_selected = (chInfo.selectedModeIndex == n);
                    if (ImGui::Selectable(chInfo.modes[n].c_str(), is_selected)) {
                        chInfo.selectedModeIndex = static_cast<uint32_t>(n);
                        chInfo.selectedMode = Utils::stringToCameraResolution(
                                chInfo.modes[chInfo.selectedModeIndex]);
                        chInfo.updateResolutionMode = true;
                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            /*
            ImGui::Dummy(ImVec2(40.0f, 10.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
            //ImGui::Checkbox("Use IMU data (Not finished)", &dev.useImuData);
            ImGui::PopStyleColor();

             */
            /*
            if (dev.useImuData) {
                if (!Utils::isInVector(dev.userRequestedSources, "IMU")) {
                    dev.userRequestedSources.emplace_back("IMU");
                    Log::Logger::getInstance()->info(("Adding IMU source to user requested sources"));
                }
            } else {
                Utils::removeFromVector(&dev.userRequestedSources, "IMU");
            }
   */
            // Check if mouse hover a window

            ImGui::Dummy(ImVec2(40.0f, 10.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
            ImGui::RadioButton("ArcBall camera", &dev.cameraType, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Arrow keys", &dev.cameraType, 1);
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            dev.resetCamera = ImGui::Button("Reset camera position");
            ImGui::PopStyleColor(2);

        }

        if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD){
            auto &chInfo = dev.channelInfo.front();
            dev.win.at(AR_PREVIEW_POINT_CLOUD).selectedSource = "Disparity Left";
            if (!Utils::isInVector(chInfo.requestedStreams, "Disparity Left")) {
                chInfo.requestedStreams.emplace_back("Disparity Left");
                Log::Logger::getInstance()->info(("Adding Disparity Left source to user requested sources"));
            }
            if (!Utils::isInVector(chInfo.requestedStreams, "Luma Rectified Left")) {
                chInfo.requestedStreams.emplace_back("Luma Rectified Left");
                Log::Logger::getInstance()->info(("Adding Luma Rectified Left source to user requested sources"));
            }
        }

        handles->disableCameraRotationFromGUI = (ImGui::IsWindowHovered() ||
                                                 ImGui::IsWindowHoveredByName("SideBar", ImGuiHoveredFlags_AnyWindow) ||
                                                 ImGui::IsAnyItemActive());

    }

    void createControlArea(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 15.0f);

        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLCoolGray);
        ImGui::SetNextWindowSize(ImVec2(handles->info->controlAreaWidth, handles->info->controlAreaHeight));
        ImGui::Begin("ControlArea", &pOpen, window_flags);


        for (auto &d: handles->devices) {
            // Create dropdown
            if (d.state == AR_STATE_ACTIVE) {

                ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Preview Control")) {

                        drawVideoPreviewGuiOverlay(handles, dev, true);
                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);

                    if (ImGui::BeginTabItem("Sensor Config")) {

                        drawVideoPreviewGuiOverlay(handles, dev, false);
                        buildConfigurationTab(handles, dev);
                        ImGui::EndTabItem();
                    }
                    /*
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Future Tab 1")) {

                        ImGui::EndTabItem();
                    }
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem("Future Tab 2")) {

                        ImGui::EndTabItem();

                    }
                     */

                    ImGui::EndTabBar();
                }
            }
        }
        ImGui::PopStyleColor();

        // Draw border between control and viewing area
        ImVec2 lineMin(handles->info->sidebarWidth + handles->info->controlAreaWidth - 3.0f, 0.0f);
        ImVec2 lineMax(lineMin.x + 3.0f, handles->info->height);
        ImGui::GetWindowDrawList()->AddRectFilled(lineMin, lineMax, ImColor(VkRender::CRLGray424Main), 0.0f,
                                                  0);
        ImGui::Dummy(ImVec2(0.0f, handles->info->height - ImGui::GetCursorPosY()));
        ImGui::End();
        ImGui::PopStyleVar(3);
    }

    void buildConfigurationTab(VkRender::GuiObjectHandles *handles, VkRender::Device &d) {
        // Exposure Tab
        {
            float textSpacing = 90.0f;
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);

            if (d.isRemoteHead) {
                ImGui::Dummy(ImVec2(0.0f, 10.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                if (ImGui::RadioButton("Head 0", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_0))
                    d.parameters.updateGuiParams = true;
                ImGui::SameLine(0, 10.0f);

                if (ImGui::RadioButton("Head 1", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_1))
                    d.parameters.updateGuiParams = true;
                ImGui::SameLine(0, 10.0f);

                if (ImGui::RadioButton("Head 2", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_2))
                    d.parameters.updateGuiParams = true;
                ImGui::SameLine(0, 10.0f);
                if (ImGui::RadioButton("Head 3", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_3))
                    d.parameters.updateGuiParams = true;
            }

            ImGui::PushFont(handles->info->font18);
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::Dummy(ImVec2(10.0f, 0.0f));
            ImGui::SameLine();
            ImGui::Text("Exposure");
            ImGui::PopFont();

            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLBlueIsh);
            ImGui::SameLine(0, 135.0f);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5.0f);
            ImGui::Text("Hold left ctrl to type in values");
            ImGui::PopStyleColor();

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
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            if (!d.parameters.ep.autoExposure) {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::HelpMarker("\n Exposure in microseconds \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Exposure:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderInt("##Exposure Value: ", reinterpret_cast<int *>(&d.parameters.ep.exposure),
                                 20, 30000);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Gain:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Gain",
                                   &d.parameters.gain, 1.68f,
                                   14.2f, "%.1f");
                d.parameters.update = ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

            } else {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::HelpMarker("\n Max exposure in microseconds \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Max Exp.:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Threshold", &d.parameters.ep.autoExposureThresh,
                                   0, 1);
                d.parameters.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                static char buf1[5] = "0";
                static char buf2[5] = "0";
                static char buf3[5] = "0";
                static char buf4[5] = "0";
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::HelpMarker("\n Set the Region Of Interest for the auto exposure in the left image \n ");
                ImGui::SameLine(0.0f, 5.0f);

                if (ImGui::Button("Set new ROI", ImVec2(80.0f, 20.0f))) {
                    try {
                        d.parameters.ep.autoExposureRoiX = std::stoi(buf1);
                        d.parameters.ep.autoExposureRoiY = std::stoi(buf2);
                        d.parameters.ep.autoExposureRoiWidth = std::stoi(buf3) - d.parameters.ep.autoExposureRoiX;
                        d.parameters.ep.autoExposureRoiHeight = std::stoi(buf4) - d.parameters.ep.autoExposureRoiY;
                        d.parameters.ep.update |= true;
                    } catch (...) {
                        Log::Logger::getInstance()->error(
                                "Failed to parse ROI input. User most likely tried to set empty parameters");
                        d.parameters.ep.update = false;
                    }
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();
                float posX = ImGui::GetCursorPosX();
                float inputWidth = 15.0f * 2.8;
                ImGui::Text("Upper left corner (x, y)");

                ImGui::SameLine(0, 15.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SetNextItemWidth(inputWidth);
                ImGui::InputText("##decimalminX", buf1, 5, ImGuiInputTextFlags_CharsDecimal);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(inputWidth);
                ImGui::InputText("##decimalminY", buf2, 5, ImGuiInputTextFlags_CharsDecimal);
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::SetCursorPosX(posX);
                ImGui::Text("Lower right corner (x, y)");
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SetNextItemWidth(inputWidth);
                ImGui::InputText("##decimalmaxX", buf3, 5, ImGuiInputTextFlags_CharsDecimal);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(inputWidth);
                ImGui::InputText("##decimalmaxY", buf4, 5, ImGuiInputTextFlags_CharsDecimal);
                ImGui::PopStyleColor();

            }
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::Dummy(ImVec2(25.0f, 0.0f));
            ImGui::SameLine();
            txt = "Gamma:";
            txtSize = ImGui::CalcTextSize(txt.c_str());
            ImGui::Text("%s", txt.c_str());
            ImGui::SameLine(0, textSpacing - txtSize.x);
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
            ImGui::SliderFloat("##Gamma",
                               &d.parameters.gamma, 1.1f,
                               2.2f, "%.2f");
            // Correct update sequence. This is because gamma and gain was part of general parameters. This will probably be redone in the future once established categories are in place
            if (d.parameters.ep.autoExposure)
                d.parameters.update = ImGui::IsItemDeactivatedAfterEdit();
            else
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::PopStyleColor();

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
                std::string txtAutoConfig = "Enable Auto:";
                ImVec2 txtSizeAutoConfig = ImGui::CalcTextSize(txtAutoConfig.c_str());
                ImGui::Text("%s", txtAutoConfig.c_str());
                ImGui::SameLine(0, textSpacing - txtSizeAutoConfig.x);
                ImGui::Checkbox("##EnableAutoWhiteBalance", &d.parameters.wb.autoWhiteBalance);
                d.parameters.wb.update = ImGui::IsItemDeactivatedAfterEdit();
                ImGui::Dummy(ImVec2(0.0f, 5.0f));

                if (!d.parameters.wb.autoWhiteBalance) {
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Red Balance:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
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
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
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
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
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
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                    ImGui::SliderInt("##DecayRateWB",
                                     reinterpret_cast<int *>(&d.parameters.wb.autoWhiteBalanceDecay), 0,
                                     20);
                    d.parameters.wb.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();
                }

            }

            // LightingParams controls
            {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("LED Control");
                ImGui::PopFont();

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                std::string txtEnableFlash = "Flash LED:";
                ImVec2 txtSizeEnableFlash = ImGui::CalcTextSize(txtEnableFlash.c_str());
                ImGui::Text("%s", txtEnableFlash.c_str());
                ImGui::SameLine(0, textSpacing - txtSizeEnableFlash.x);
                ImGui::Checkbox("##Enable Lights", &d.parameters.light.flashing);
                d.parameters.light.update = ImGui::IsItemDeactivatedAfterEdit();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Duty Cycle :";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Duty_Cycle",
                                   &d.parameters.light.dutyCycle, 0,
                                   100,
                                   "%.0f"); // showing 0 float precision not using int cause underlying libmultisense is a float
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                /*
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Light Index:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderInt("##LightSelection",
                                 reinterpret_cast<int *>(&d.parameters.light.selection), -1,
                                 3);
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
*/
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::HelpMarker("\n Light pulses per exposure \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Light Pulses:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Pulses",
                                   reinterpret_cast<float *>(&d.parameters.light.numLightPulses), 0,
                                   60, "%.1f");
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::HelpMarker("\n LED startup time in milliseconds \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Startup Time:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::SetNextItemWidth(handles->info->controlAreaWidth - 72.0f - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Startup Time",
                                   reinterpret_cast<float *>(&d.parameters.light.startupTime), 0,
                                   60, "%.1f");
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }
            // Additional Params
            {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("General");
                ImGui::PopFont();

                // HDR
                /*
                {
                    ImGui::Dummy(ImVec2(0.0f, 15.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "HDR";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::Checkbox("##HDREnable", &d.parameters.hdrEnabled);
                    d.parameters.update = ImGui::IsItemDeactivatedAfterEdit();
                }
                */

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Framerate:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Framerate",
                                   &d.parameters.fps, 1,
                                   30, "%.1f");
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Stereo Filter:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SliderFloat("##Stereo",
                                   &d.parameters.stereoPostFilterStrength, 0.0f,
                                   1.0f, "%.1f");
                d.parameters.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }

            // Calibration
            {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("Calibration");
                ImGui::PopFont();

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Save the current camera calibration to directory";
                ImGui::Text("%s", txt.c_str());

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Location:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImVec2 btnSize(100.0f, 20.0f);
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - (btnSize.x) - 35.0f);
                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::CRLTextWhiteDisabled);

#ifdef WIN32
                std::string hint = "C:\\Path\\To\\dir";
#else
                std::string hint = "/Path/To/dir";
#endif
                ImGui::CustomInputTextWithHint("##SaveLocation", hint.c_str(), &d.parameters.calib.saveCalibrationPath,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();

                ImGui::SameLine();


                if (ImGui::Button("Choose Dir", btnSize))
                    saveCalibrationDialog.OpenDialog("ChooseDirDlgKey", "Choose save location", nullptr,
                                                     ".");
                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLDarkGray425);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                if (saveCalibrationDialog.Display("ChooseDirDlgKey", 0, ImVec2(600.0f, 400.0f),
                                                  ImVec2(1200.0f, 1000.0f))) {
                    // action if OK
                    if (saveCalibrationDialog.IsOk()) {
                        std::string filePathName = saveCalibrationDialog.GetFilePathName();
                        d.parameters.calib.saveCalibrationPath = filePathName;
                        // action
                    }
                    // close
                    saveCalibrationDialog.Close();
                }
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);

                if (d.parameters.calib.saveCalibrationPath == "Path/To/Dir") {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::TextColorGray);
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::TextColorGray);
                }
                d.parameters.calib.save = ImGui::Button("Get Current Calibration");

                if (d.parameters.calib.saveCalibrationPath == "Path/To/Dir") {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleColor(2);
                }
                ImGui::PopStyleColor();

                if (d.parameters.calib.save) {
                    showSavedTimer = std::chrono::steady_clock::now();
                }
                auto time = std::chrono::steady_clock::now();
                float threeSeconds = 3.0f;
                std::chrono::duration<float> time_span =
                        std::chrono::duration_cast<std::chrono::duration<float>>(time - showSavedTimer);
                if (time_span.count() < threeSeconds) {
                    ImGui::SameLine();
                    if (d.parameters.calib.saveFailed) {
                        ImGui::Text("Saved!");
                    } else {
                        ImGui::Text("Failed to save calibration");
                    }

                }

                ImGui::Dummy(ImVec2(0.0f, 10.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Set new camera calibration";
                ImGui::Text("%s", txt.c_str());

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Intrinsics:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - (btnSize.x) - 35.0f);
                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::CRLTextWhiteDisabled);

#ifdef WIN32
                hint = "C:\\Path\\To\\dir";
#else
                hint = "/Path/To/dir";
#endif
                ImGui::CustomInputTextWithHint("##IntrinsicsLocation", hint.c_str(),
                                               &d.parameters.calib.intrinsicsFilePath,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();
                ImGui::SameLine();


                if (ImGui::Button("Choose File##1", btnSize))
                    chooseIntrinsicsDialog.OpenDialog("ChooseFileDlgKey", "Choose intrinsics .yml file", ".yml",
                                                      ".");
                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLDarkGray425);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                if (chooseIntrinsicsDialog.Display("ChooseFileDlgKey", 0, ImVec2(600.0f, 400.0f),
                                                   ImVec2(1200.0f, 1000.0f))) {
                    // action if OK
                    if (chooseIntrinsicsDialog.IsOk()) {
                        std::string filePathName = chooseIntrinsicsDialog.GetFilePathName();
                        d.parameters.calib.intrinsicsFilePath = filePathName;
                        // action
                    }
                    // close
                    chooseIntrinsicsDialog.Close();
                }
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Extrinsics:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - (btnSize.x) - 35.0f);
                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::CRLTextWhiteDisabled);

#ifdef WIN32
                hint = "C:\\Path\\To\\file";
#else
                hint = "/Path/To/file";
#endif
                ImGui::CustomInputTextWithHint("##ExtrinsicsLocation", hint.c_str(),
                                               &d.parameters.calib.extrinsicsFilePath,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();
                ImGui::SameLine();
                if (ImGui::Button("Choose File##2", btnSize))
                    chooseExtrinsicsDialog.OpenDialog("ChooseFileDlgKey", "Choose extrinsics .yml file", ".yml",
                                                      ".");
                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLDarkGray425);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                if (chooseExtrinsicsDialog.Display("ChooseFileDlgKey", 0, ImVec2(600.0f, 400.0f),
                                                   ImVec2(1200.0f, 1000.0f))) {
                    // action if OK
                    if (chooseExtrinsicsDialog.IsOk()) {
                        std::string filePathName = chooseExtrinsicsDialog.GetFilePathName();
                        d.parameters.calib.extrinsicsFilePath = filePathName;
                        // action
                    }
                    // close
                    chooseExtrinsicsDialog.Close();
                }
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);

                if (d.parameters.calib.intrinsicsFilePath == "Path/To/Intrinsics.yml" ||
                    d.parameters.calib.extrinsicsFilePath == "Path/To/Extrinsics.yml") {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::TextColorGray);
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::TextColorGray);
                }
                if (ImGui::Button("Set New Calibration")) {
                    ImGui::OpenPopup("Overwrite calibration?");

                }

                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 5.0f));
                if (ImGui::BeginPopupModal("Overwrite calibration?", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                    ImGui::Text(
                            "Setting a new calibration will overwrite the current setting.\nThis operation cannot be undone!\n\n");
                    ImGui::Separator();

                    if (ImGui::Button("OK", ImVec2(120, 0))) {
                        d.parameters.calib.update = true;
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SetItemDefaultFocus();
                    ImGui::SameLine();


                    ImGui::SetCursorPosX( ImGui::GetWindowWidth() - ImGui::GetCursorPosX() + 8.0f);
                    if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
                    ImGui::EndPopup();
                }
                ImGui::PopStyleVar(2);

                if (d.parameters.calib.intrinsicsFilePath == "Path/To/Intrinsics.yml" ||
                    d.parameters.calib.extrinsicsFilePath == "Path/To/Extrinsics.yml") {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleColor(2);
                }
                ImGui::PopStyleColor();

                if (d.parameters.calib.update) {
                    showSetTimer = std::chrono::steady_clock::now();
                }
                time = std::chrono::steady_clock::now();
                threeSeconds = 3.0f;
                time_span = std::chrono::duration_cast<std::chrono::duration<float>>(time - showSetTimer);
                if (time_span.count() < threeSeconds) {
                    ImGui::SameLine();
                    if (d.parameters.calib.updateFailed) {
                        ImGui::Text("Set calibration!");
                    } else {
                        ImGui::Text("Failed to set calibration...");
                    }

                }


            }
            ImGui::PopStyleColor();
        }
    }

};

#endif //MULTISENSE_INTERACTIONMENU_H
