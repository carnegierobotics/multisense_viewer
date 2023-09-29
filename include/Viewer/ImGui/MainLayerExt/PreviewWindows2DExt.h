/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/Layer.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-4-19, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_2D_Preview_Windows_H
#define MULTISENSE_VIEWER_2D_Preview_Windows_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/**
 * @brief A UI Layer drawn by \refitem GuiManager.
 * To add an additional UI layer see \refitem LayerExample.
 */
class PreviewWindows2DExt : public VkRender::Layer {

public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        for (auto &dev: handles->devices) {
            if (dev.state != CRL_STATE_ACTIVE)
                continue;

            ImGui::SetNextWindowPos(handles->info->viewingAreaWindowPos, ImGuiCond_Always);
            handles->info->viewingAreaWidth = handles->info->viewingAreaWidth;

            ImGui::SetNextWindowSize(ImVec2(handles->info->viewingAreaWidth, handles->info->height),
                                     ImGuiCond_Always);

            bool pOpen = true;

            ImGui::PushStyleColor(ImGuiCol_WindowBg,
                                  ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); // Set the window background color to transparent
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                                ImVec2(0.0f, 0.0f)); // Set the window background color to transparent
            ImGui::Begin("View Area", &pOpen,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                         ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
            handles->info->isViewingAreaHovered = ImGui::IsWindowHovered(
                    ImGuiHoveredFlags_RootAndChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup);
            ImGui::PopStyleColor();
            ImGui::PopStyleVar();
            if (dev.selectedPreviewTab == CRL_TAB_2D_PREVIEW)
                drawVideoPreviewGuiOverlay(handles, dev);
            ImGui::End(); // End the empty view area window

            // The top left corner of the ImGui window that encapsulates the quad with the texture playing.
            bool hoveringPreviewWindows = ImGui::IsWindowHoveredByName("View Area 0", ImGuiHoveredFlags_AnyWindow) ||
                                          ImGui::IsWindowHoveredByName("View Area 1", ImGuiHoveredFlags_AnyWindow);
            handles->scroll = 0.0f;
            bool hoveringPopupWindows = (dev.win[(StreamWindowIndex) 0].isHovered ||
                                         dev.win[(StreamWindowIndex) 1].isHovered);
            if (handles->info->isViewingAreaHovered && dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE &&
                !handles->info->hoverState) {
                handles->accumulatedActiveScroll -= ImGui::GetIO().MouseWheel * 100.0f;
                handles->scroll = ImGui::GetIO().MouseWheel * 100.0f;

                float diff = 0.0f;
                if (handles->accumulatedActiveScroll > handles->maxScroll) {
                    diff = handles->accumulatedActiveScroll - handles->maxScroll;
                    handles->accumulatedActiveScroll = handles->maxScroll;
                }
                if (handles->accumulatedActiveScroll < handles->minScroll) {
                    diff = handles->accumulatedActiveScroll - handles->minScroll;
                    handles->accumulatedActiveScroll = handles->minScroll;
                }
                handles->scroll += diff;

            }
        }

    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }

    void drawVideoPreviewGuiOverlay(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        if (dev.layout != CRL_PREVIEW_LAYOUT_NONE) {
            createWindowPreviews(handles, dev);
        }

        // Some Point cloud enable/disable sources logic
        // Remove these two sources if we switch between 3D to 2D and we are not using the sources in 2D
        std::vector<std::string> pointCloudSources({"Disparity Left", "Luma Rectified Left"});
        // Disable IMU as well in 2D
        auto &chInfo = dev.channelInfo.front();
        for (const auto &source: pointCloudSources) {
            bool inUse = false;
            // Loop over all previews and check their source.
            // If it matches either point cloud source then it means it is in use
            for (const auto &preview: dev.win) {
                if (preview.second.selectedSource == source && preview.first != CRL_PREVIEW_POINT_CLOUD)
                    inUse = true;
            }
            if (!inUse && Utils::isInVector(chInfo.requestedStreams, source)) {
                Utils::removeFromVector(&chInfo.requestedStreams, source);
                Log::Logger::getInstance()->info(
                        "Removed {} from user requested sources because it is not in use anymore", source);
            }
        }

    }

    void createWindowPreviews(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        handles->info->viewingAreaWidth =
                handles->info->width - handles->info->sidebarWidth - handles->info->controlAreaWidth;

        int cols = 0, rows = 0;
        switch (dev.layout) {
            case CRL_PREVIEW_LAYOUT_NONE:
                break;
            case CRL_PREVIEW_LAYOUT_SINGLE:
                cols = 1;
                rows = 1;
                break;
            case CRL_PREVIEW_LAYOUT_DOUBLE:
                rows = 2;
                cols = 1;
                break;
            case CRL_PREVIEW_LAYOUT_DOUBLE_SIDE_BY_SIDE:
                rows = 1;
                cols = 2;
                break;
            case CRL_PREVIEW_LAYOUT_QUAD:
                cols = 2;
                rows = 2;
                break;
            case CRL_PREVIEW_LAYOUT_NINE:
                cols = 3;
                rows = 3;
                break;
        }

        int index = 0;
        handles->info->hoverState = false;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                std::string windowName = std::string("View Area ") + std::to_string(index);
                auto &window = dev.win[(StreamWindowIndex) index];
                window.name = windowName;

                float newWidth = (handles->info->width - (640 + (5.0f + (5.0f * (float) cols)))) / (float) cols;
                if (dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE)
                    newWidth -= 50.0f * (handles->info->width / 1280);

                float newHeight = newWidth * (10.0f / 16.0f); // aspect ratio 16:10 of camera images

                handles->info->viewAreaElementSizeX = newWidth;
                handles->info->viewAreaElementSizeY = newHeight;
                handles->info->previewBorderPadding = 35.0f * (handles->info->width / 1280);


                float offsetX;
                if (dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE || dev.layout == CRL_PREVIEW_LAYOUT_SINGLE) {
                    offsetX = (handles->info->controlAreaWidth + handles->info->sidebarWidth +
                               ((handles->info->viewingAreaWidth - newWidth) / 2));
                } else
                    offsetX = (handles->info->controlAreaWidth + handles->info->sidebarWidth + 5.0f);

                float viewAreaElementPosX = offsetX + ((float) col * (newWidth + 10.0f));

                ImVec2 windowSize = ImVec2(handles->info->viewAreaElementSizeX, handles->info->viewAreaElementSizeY);

                /*ImGui::SetNextWindowSize(
                        windowSize,
                        ImGuiCond_Always);
*/
                dev.win.at((StreamWindowIndex) index).row = float(row);
                dev.win.at((StreamWindowIndex) index).col = float(col);
                // Calculate window m_Position
                float viewAreaElementPosY =
                        handles->info->tabAreaHeight + ((float) row * (handles->info->viewAreaElementSizeY + 10.0f));

                if (dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE) {
                    viewAreaElementPosY = viewAreaElementPosY + handles->accumulatedActiveScroll;
                }
                dev.win.at((StreamWindowIndex) index).xPixelStartPos = viewAreaElementPosX;
                dev.win.at((StreamWindowIndex) index).yPixelStartPos = viewAreaElementPosY;
                ImVec2 childPos = ImVec2(viewAreaElementPosX, viewAreaElementPosY);
                //ImGui::SetNextWindowPos(childPos,ImGuiCond_Always);
                ImGui::SetCursorScreenPos(childPos);
                static bool open = true;
                ImGuiWindowFlags window_flags =
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus;

                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                //ImGui::Begin(windowName.c_str(), &open, window_flags);
                ImGui::BeginChild(windowName.c_str(), windowSize, false, ImGuiWindowFlags_NoBringToFrontOnFocus);

                window.isHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup);

                // The colored bars around the preview window is made up of rectangles
                // Top bar
                ImVec2 topBarRectMin(viewAreaElementPosX, viewAreaElementPosY);
                ImVec2 topBarRectMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                                     viewAreaElementPosY + (handles->info->previewBorderPadding / 2));
                ImGui::GetWindowDrawList()->AddRectFilled(topBarRectMin, topBarRectMax,
                                                          ImColor(VkRender::Colors::CRLBlueIsh),
                                                          0.0f,
                                                          0);

                // left bar

                ImVec2 leftBarMin(viewAreaElementPosX, topBarRectMax.y);
                ImVec2 leftBarMax(viewAreaElementPosX + (handles->info->previewBorderPadding / 2),
                                  topBarRectMax.y + handles->info->viewAreaElementSizeY -
                                  (handles->info->previewBorderPadding * 0.5f));

                ImGui::GetWindowDrawList()->AddRectFilled(leftBarMin, leftBarMax,
                                                          ImColor(VkRender::Colors::CRLGray424Main),
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

                ImGui::GetWindowDrawList()->AddRectFilled(rightBarMin, rightBarMax,
                                                          ImColor(VkRender::Colors::CRLGray424Main),
                                                          0.0f,
                                                          0);

                // Bottom bar
                ImVec2 bottomBarMin(viewAreaElementPosX, topBarRectMin.y + handles->info->viewAreaElementSizeY -
                                                         (handles->info->previewBorderPadding / 2.0f));
                ImVec2 bottomBarMax(viewAreaElementPosX + handles->info->viewAreaElementSizeX,
                                    topBarRectMin.y + handles->info->viewAreaElementSizeY +
                                    (handles->info->previewBorderPadding / 2.0f));

                ImGui::GetWindowDrawList()->AddRectFilled(bottomBarMin, bottomBarMax,
                                                          ImColor(VkRender::Colors::CRLGray424Main),
                                                          0.0f,
                                                          0);

                // Min Y and Min Y is top left corner

                ImGui::SetCursorScreenPos(ImVec2(topBarRectMin.x + 20.0f, topBarRectMin.y + 2.0f));
                // Also only if a window is hovered

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 0));

                if (window.isHovered) {
                    // Offsset cursor positions.
                    switch (Utils::CRLSourceToTextureType(dev.win.at((StreamWindowIndex) index).selectedSource)) {
                        case CRL_GRAYSCALE_IMAGE:
                            ImGui::Text("(%d, %d) %d", dev.pixelInfoZoomed[(StreamWindowIndex) index].x,
                                        dev.pixelInfoZoomed[(StreamWindowIndex) index].y,
                                        dev.pixelInfoZoomed[(StreamWindowIndex) index].intensity);
                            break;
                        case CRL_DISPARITY_IMAGE:
                            ImGui::Text("(%d, %d) %.2f m", dev.pixelInfoZoomed[(StreamWindowIndex) index].x,
                                        dev.pixelInfoZoomed[(StreamWindowIndex) index].y,
                                        dev.pixelInfoZoomed[(StreamWindowIndex) index].depth);
                            break;
                        default:
                            ImGui::Text("(%d, %d)", dev.pixelInfoZoomed[(StreamWindowIndex) index].x,
                                        dev.pixelInfoZoomed[(StreamWindowIndex) index].y);
                    }

                }
                ImGui::SetCursorScreenPos(ImVec2(topBarRectMin.x + 20.0f, bottomBarMin.y + 2.0f));
                ImGui::Text("Lock zoom (Right click): ");
                ImGui::SameLine();
                bool lock = !window.enableZoom;
                if (ImGui::Checkbox(("##enableZoom" + std::to_string(index)).c_str(), &lock)) {
                    handles->usageMonitor->userClickAction("enableZoom", "Checkbox", ImGui::GetCurrentWindow()->Name);

                }
                ImGui::SameLine();

                std::string btnText = "Image Effects";
                float btnHeight = topBarRectMax.y - topBarRectMin.y;
                ImVec2 btnSize = ImGui::CalcTextSize(btnText.c_str());
                ImGui::SameLine(0, handles->info->viewAreaElementSizeX - ImGui::GetCursorPosX() - btnSize.x - 60.0f);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f);
                float btnWidth = rightBarMin.x - ImGui::GetCursorScreenPos().x;
                if (ImGui::Button(btnText.c_str(), ImVec2(btnWidth, btnHeight))) {
                    ImGui::OpenPopup(("image effect " + std::to_string(index)).c_str());
                    handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

                }
                ImGui::PopStyleVar(); // FramePadding

                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 10.0f));


                // Specific case for scrolled windows and popup position
                if (dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE) {
                    window.popupPosition.y -= handles->scroll;
                }
                if (dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE) {
                    ImVec2 pos = window.popupPosition;
                    if (window.popupPosition.x == 0.0f) {
                        pos.x = handles->mouse->pos.x;
                        pos.y = handles->mouse->pos.y;
                    }
                    if ((window.popupPosition.x + window.popupWindowSize.x) > handles->info->width) {
                        window.popupPosition.x = window.popupPosition.x -
                                                 (window.popupPosition.x + window.popupWindowSize.x + 20.0f -
                                                  handles->info->width);
                        pos = window.popupPosition;
                    }
                    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
                }

                if (ImGui::BeginPopup(("image effect " + std::to_string(index)).c_str())) {
                    if (dev.layout == CRL_PREVIEW_LAYOUT_DOUBLE) {
                        window.popupPosition = ImGui::GetWindowPos();
                        window.popupWindowSize = ImGui::GetWindowSize();
                    }
                    float textSpacing = 90.0f;
                    { // ROW 1
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
                        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                        ImGui::SameLine();
                        ImGui::HelpMarker(
                                "Hover mouse of preview window to use keyboard shortcuts to enable/disable certain effects");
                        // if start then show gif spinner
                        ImGui::PopStyleColor();
                        ImGui::PopStyleVar();
                        ImGui::SameLine(0, 5.0f);
                        ImGui::Text("Shortcut");
                        ImGui::PopStyleVar();
                        ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("Shortcut").x);
                        std::string txt = "Effect";
                        ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x - ImGui::CalcTextSize("Option").x / 2.0f);
                        ImGui::Text("Option");
                    }

                    ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                    ImGui::SameLine();
                    ImGui::Text("i");
                    ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("i").x);
                    std::string txt = "Interpolate:";
                    ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    if (ImGui::Checkbox(("##interpolate" + std::to_string(index)).c_str(),
                                        &window.effects.interpolation)) {
                        handles->usageMonitor->userClickAction("interpolate", "Checkbox",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    // Cursor zoom
                    ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                    ImGui::SameLine();
                    ImGui::Text("z");
                    ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("z").x);
                    txt = "Cursor Zoom:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    if (ImGui::Checkbox(("##zoom mode" + std::to_string(index)).c_str(),
                                        &window.effects.magnifyZoomMode)) {
                        handles->usageMonitor->userClickAction("zoom mode", "Checkbox",
                                                               ImGui::GetCurrentWindow()->Name);
                    }

                    bool isColorImageSelected =
                            Utils::CRLSourceToTextureType(dev.win.at((StreamWindowIndex) index).selectedSource) ==
                            CRL_COLOR_IMAGE_YUV420;


                    bool isDisparitySelected =
                            Utils::CRLSourceToTextureType(dev.win.at((StreamWindowIndex) index).selectedSource) ==
                            CRL_DISPARITY_IMAGE;

                    if (isColorImageSelected && !VkRender::RendererConfig::getInstance().hasEnabledExtension(
                            VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME) || isDisparitySelected) {

                    } else {

                        // Edge detection
                        ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                        ImGui::SameLine();
                        ImGui::Text("1");
                        ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("1").x);
                        txt = "Edge:";
                        txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x);
                        if (ImGui::Checkbox(("##edge filter" + std::to_string(index)).c_str(),
                                            &window.effects.edgeDetection)) {
                            handles->usageMonitor->userClickAction("Edge filter", "Checkbox",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }

                        // Blurring
                        ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                        ImGui::SameLine();
                        ImGui::Text("2");
                        ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("2").x);
                        txt = "Blur:";
                        txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x);
                        if (ImGui::Checkbox(("##blur filter" + std::to_string(index)).c_str(),
                                            &window.effects.blur)) {
                            handles->usageMonitor->userClickAction("Blur filter", "Checkbox",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }


                        ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                        ImGui::SameLine();
                        ImGui::Text("3");
                        ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("3").x);
                        txt = "Emboss:";
                        txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x);
                        if (ImGui::Checkbox(("##Emboss filter" + std::to_string(index)).c_str(),
                                            &window.effects.emboss)) {
                            handles->usageMonitor->userClickAction("Emboss filter", "Checkbox",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }


                        ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                        ImGui::SameLine();
                        ImGui::Text("4");
                        ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("4").x);
                        txt = "Sharpen:";
                        txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x);
                        if (ImGui::Checkbox(("##Sharpen filter" + std::to_string(index)).c_str(),
                                            &window.effects.sharpening)) {
                            handles->usageMonitor->userClickAction("Sharpen filter", "Checkbox",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }
                    }

                    if (isDisparitySelected) {
                        // Color map
                        {
                            ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                            ImGui::SameLine();
                            ImGui::Text("m");
                            ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("m").x);
                            txt = "Color map:";
                            txtSize = ImGui::CalcTextSize(txt.c_str());
                            ImGui::Text("%s", txt.c_str());
                            ImGui::SameLine(0, textSpacing - txtSize.x);
                            if (ImGui::Checkbox(("##useDepthColorMap" + std::to_string(index)).c_str(),
                                                &window.effects.depthColorMap)) {
                                handles->usageMonitor->userClickAction("useDepthColorMap", "Checkbox",
                                                                       ImGui::GetCurrentWindow()->Name);
                            }
                        }

                        // normalize
                        {
                            ImGui::Dummy(ImVec2((ImGui::CalcTextSize("(?)Shortcut").x / 2.0f), 0.0f));
                            ImGui::SameLine();
                            ImGui::Text("n");
                            ImGui::SameLine(0, 40.0f - ImGui::CalcTextSize("n").x);
                            txt = "Normalize:";
                            txtSize = ImGui::CalcTextSize(txt.c_str());
                            ImGui::Text("%s", txt.c_str());
                            ImGui::SameLine(0, textSpacing - txtSize.x);
                            if (ImGui::Checkbox(("##normalize" + std::to_string(index)).c_str(),
                                                &window.effects.normalize)) {
                                handles->usageMonitor->userClickAction("normalize", "Checkbox",
                                                                       ImGui::GetCurrentWindow()->Name);
                            }
                        }
                    }

                    //window.isHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
                    ImGui::EndPopup();
                }
                ImGui::PopStyleVar(); // window padding


                // Max X and Min Y is top right corner
                ImGui::SetCursorScreenPos(ImVec2(topBarRectMax.x - 235.0f, topBarRectMin.y));
                ImGui::SetNextItemWidth(80.0f);

                if (dev.isRemoteHead) {
                    ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLBlueIsh);
                    std::string label = window.availableRemoteHeads[Utils::getIndexOf(window.availableRemoteHeads,
                                                                                      std::to_string(
                                                                                              window.selectedRemoteHeadIndex +
                                                                                              1))];
                    std::string comboLabel = "##RemoteHeadSelection" + std::to_string(index);
                    if (ImGui::BeginCombo(comboLabel.c_str(), label.c_str(),
                                          ImGuiComboFlags_HeightLarge)) {
                        for (size_t n = 0; n < window.availableRemoteHeads.size(); n++) {
                            const bool is_selected = (window.selectedRemoteHeadIndex ==
                                                      (crl::multisense::RemoteHeadChannel) n);
                            if (ImGui::Selectable(window.availableRemoteHeads[n].c_str(), is_selected)) {


                                // Disable the previously enabled source if not in use and update the selected source tab
                                if (window.selectedSource != "Idle") {
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
                                        window.selectedSource = window.availableSources[0]; // 0th index is always "Idle"
                                        window.selectedSourceIndex = 0;
                                    }
                                    // If its in use, but we don't want to disable it. Just reset the source name to "Idle"
                                    if (inUse && !sourceInUse.empty()) {
                                        window.selectedSource = window.availableSources[0]; // 0th index is always "Idle"
                                        window.selectedSourceIndex = 0;
                                    }
                                }

                                window.selectedRemoteHeadIndex = (crl::multisense::RemoteHeadChannel) std::stoi(
                                        window.availableRemoteHeads[n]) - (crl::multisense::RemoteHeadChannel) 1;
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
                window.availableSources = dev.channelInfo[(crl::multisense::RemoteHeadChannel) std::stoi(
                        window.availableRemoteHeads[
                                Utils::getIndexOf(window.availableRemoteHeads,
                                                  std::to_string(window.selectedRemoteHeadIndex + 1))]) -
                                                          1].availableSources;

                Log::Logger::getInstance()->traceWithFrequency("Displayed_Available_Sources", 60 * 10,
                                                               "Presented sources to user: ");
                for (const auto &src: window.availableSources) {
                    Log::Logger::getInstance()->traceWithFrequency("Tag:" + src, 60 * 10, "{}", src);
                }

                ImGui::SetCursorScreenPos(ImVec2(topBarRectMax.x - 150.0f, topBarRectMin.y));
                ImGui::SetNextItemWidth(150.0f);
                std::string srcLabel = "##Source" + std::to_string(index);
                std::string previewValue;
                ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLBlueIsh);
                std::string colorSources = "Color Aux";
                std::string auxLumaSources = "Luma Aux";
                std::string colorRectifiedSources = "Color Rectified Aux";
                std::string auxLumaRectifiedSources = "Luma Rectified Aux";
                if (ImGui::BeginCombo(srcLabel.c_str(), window.selectedSource.c_str(),
                                      ImGuiComboFlags_HeightLarge)) {
                    for (size_t n = 0; n < window.availableSources.size(); n++) {
                        const bool is_selected = (window.selectedSourceIndex == n);
                        if (ImGui::Selectable(window.availableSources[n].c_str(), is_selected)) {

                            if (window.selectedSource != "Idle") {
                                bool inUse = false;
                                bool stopColor = false;
                                for (const auto &otherWindow: dev.win) {
                                    if (otherWindow.second.selectedSource == window.selectedSource &&
                                        (int) otherWindow.first != index &&
                                        otherWindow.second.selectedRemoteHeadIndex == window.selectedRemoteHeadIndex) {
                                        inUse = true;
                                    }

                                    // If a color source is active in another window and our selected source is a aux *luma source then do nothing
                                    if (otherWindow.first != index &&
                                        (colorSources == otherWindow.second.selectedSource) &&
                                        (auxLumaSources == window.selectedSource)) {
                                        inUse = true;
                                    }
                                    // If a rectified color source is active in another window and our selected source is a aux *luma source then do nothing
                                    if (otherWindow.first != index &&
                                        (colorRectifiedSources == otherWindow.second.selectedSource) &&
                                        (auxLumaRectifiedSources == window.selectedSource)) {
                                        inUse = true;
                                    }
                                    // If a color source is closed but we have a luma source active then only stop the color source
                                    if (otherWindow.first != index &&
                                        (auxLumaSources == otherWindow.second.selectedSource) &&
                                        (colorSources == window.selectedSource)) {
                                        stopColor = true;
                                    }

                                    if (otherWindow.first != index &&
                                        (auxLumaRectifiedSources == otherWindow.second.selectedSource) &&
                                        (colorRectifiedSources == window.selectedSource)) {
                                        stopColor = true;
                                    }

                                    // It's in use if we have color aux running but we are disabling luma aux
                                }

                                if (!inUse && Utils::removeFromVector(
                                        &dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                        window.selectedSource)) {
                                    Log::Logger::getInstance()->info("Removed source '{}' from user requested sources",
                                                                     window.selectedSource);
                                    if (window.selectedSource == "Color Aux" && stopColor) {
                                        Utils::removeFromVector(
                                                &dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                                "Luma Aux");
                                    }
                                    if (window.selectedSource == "Color Rectified Aux" && stopColor) {
                                        Utils::removeFromVector(
                                                &dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                                "Luma Rectified Aux");
                                    }
                                }


                            }


                            window.selectedSourceIndex = static_cast<uint32_t>(n);
                            window.selectedSource = window.availableSources[window.selectedSourceIndex];
                            Log::Logger::getInstance()->info("Selected source '{}' for preview {},",
                                                             window.selectedSource, index);
                            handles->usageMonitor->userClickAction(window.selectedSource, srcLabel,
                                                                   ImGui::GetCurrentWindow()->Name);

                            if (!Utils::isInVector(dev.channelInfo[window.selectedRemoteHeadIndex].enabledStreams,
                                                   window.selectedSource) && window.selectedSource != "Idle") {
                                dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams.emplace_back(
                                        window.selectedSource);
                                Log::Logger::getInstance()->info(
                                        "Added source '{}' from head {} to user requested sources",
                                        window.selectedSource, window.selectedRemoteHeadIndex);

                                if (window.selectedSource == "Color Aux" &&
                                    !Utils::isInVector(dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                                       "Luma Aux")) {
                                    dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams.emplace_back(
                                            "Luma Aux");
                                }

                                if (window.selectedSource == "Color Rectified Aux" &&
                                    !Utils::isInVector(dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams,
                                                       "Luma Rectified Aux")) {
                                    dev.channelInfo[window.selectedRemoteHeadIndex].requestedStreams.emplace_back(
                                            "Luma Rectified Aux");
                                }
                            }

                        }
                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                handles->info->hoverState |= ImGui::IsWindowHovered(
                        ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup);
                ImGui::PopStyleColor();
                ImGui::PopStyleColor(); // PopupBg
                /** Color rest of area in the background color exluding previews**/
                ImGui::EndChild();
                if (window.isHovered) {
                    if (handles->mouse->right && handles->mouse->action == GLFW_PRESS)
                        window.enableZoom = !window.enableZoom;

                    if (handles->input->getButtonDown(GLFW_KEY_I)) {
                        window.effects.interpolation = !window.effects.interpolation;
                        Log::Logger::getInstance()->info("User pressed key I for: {}", window.name);
                        handles->usageMonitor->userClickAction("I", "keyboard_press",
                                                               ImGui::GetCurrentWindow()->Name);

                    }
                    if (window.selectedSource != "Disparity Left") {
                        if (handles->input->getButtonDown(GLFW_KEY_1)) {
                            window.effects.edgeDetection = !window.effects.edgeDetection;
                            Log::Logger::getInstance()->info("User pressed key 1 for: {}", window.name);
                            handles->usageMonitor->userClickAction("1", "keyboard_press",
                                                                   ImGui::GetCurrentWindow()->Name);

                            window.effects.blur = false;
                            window.effects.emboss = false;
                            window.effects.sharpening = false;
                        }

                        if (handles->input->getButtonDown(GLFW_KEY_2)) {
                            window.effects.blur = !window.effects.blur;
                            Log::Logger::getInstance()->info("User pressed key 2 for: {}", window.name);
                            handles->usageMonitor->userClickAction("2", "keyboard_press",
                                                                   ImGui::GetCurrentWindow()->Name);
                            window.effects.emboss = false;
                            window.effects.sharpening = false;
                            window.effects.edgeDetection = false;
                        }

                        if (handles->input->getButtonDown(GLFW_KEY_3)) {
                            window.effects.emboss = !window.effects.emboss;
                            Log::Logger::getInstance()->info("User pressed key 3 for: {}", window.name);
                            handles->usageMonitor->userClickAction("3", "keyboard_press",
                                                                   ImGui::GetCurrentWindow()->Name);
                            window.effects.blur = false;
                            window.effects.sharpening = false;
                            window.effects.edgeDetection = false;
                        }

                        if (handles->input->getButtonDown(GLFW_KEY_4)) {
                            window.effects.sharpening = !window.effects.sharpening;
                            Log::Logger::getInstance()->info("User pressed key 4 for: {}", window.name);
                            handles->usageMonitor->userClickAction("4", "keyboard_press",
                                                                   ImGui::GetCurrentWindow()->Name);
                            window.effects.blur = false;
                            window.effects.emboss = false;
                            window.effects.edgeDetection = false;
                        }
                    } else {
                        window.effects.blur = false;
                        window.effects.emboss = false;
                        window.effects.sharpening = false;
                        window.effects.edgeDetection = false;
                    }

                    if (handles->input->getButtonDown(GLFW_KEY_Z)) {
                        window.effects.magnifyZoomMode = !window.effects.magnifyZoomMode;
                        Log::Logger::getInstance()->info("User pressed key Z for: {}", window.name);
                        handles->usageMonitor->userClickAction("Z", "keyboard_press", ImGui::GetCurrentWindow()->Name);
                    }
                    if (handles->input->getButtonDown(GLFW_KEY_M)) {
                        window.effects.depthColorMap = !window.effects.depthColorMap;
                        Log::Logger::getInstance()->info("User pressed key M for: {}", window.name);
                        handles->usageMonitor->userClickAction("M", "keyboard_press", ImGui::GetCurrentWindow()->Name);
                    }
                    if (handles->input->getButtonDown(GLFW_KEY_N)) {
                        window.effects.normalize = !window.effects.normalize;
                        Log::Logger::getInstance()->info("User pressed key N for: {}", window.name);
                        handles->usageMonitor->userClickAction("N", "keyboard_press", ImGui::GetCurrentWindow()->Name);

                    }

                    if (window.enableZoom) {
                        handles->previewZoom[static_cast<StreamWindowIndex>(index)] += ImGui::GetIO().MouseWheel / 5.0f;

                        if (handles->previewZoom[static_cast<StreamWindowIndex>(index)] > handles->maxZoom) {
                            handles->previewZoom[static_cast<StreamWindowIndex>(index)] = handles->maxZoom;
                        }
                        if (handles->previewZoom[static_cast<StreamWindowIndex>(index)] < handles->minZoom) {
                            handles->previewZoom[static_cast<StreamWindowIndex>(index)] = handles->minZoom;
                        }
                    }
                }
                index++;
            }
        }
    }

};


#endif //ControlAreaExtension
