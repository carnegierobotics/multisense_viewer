/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/InteractionMenu.h
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
 *   2022-5-5, mgjerde@carnegierobotics.com, Created file.
 **/


#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H
#ifdef WIN32
    #ifdef APIENTRY
        #undef APIENTRY
    #endif
#endif
#include <GLFW/glfw3.h>
#include <filesystem>

#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/ImGui/MainLayerExt/ControlArea2DExt.h"
#include "Viewer/ImGui/MainLayerExt/PreviewWindows2DExt.h"
#include "Viewer/ImGui/MainLayerExt/Preview3DExt.h"
#include "Viewer/ImGui/MainLayerExt/SensorConfigurationExt.h"

#ifdef WIN32
#else

#include <unistd.h>

#endif


class MainLayer : public VkRender::Layer {
private:
    std::unique_ptr<ControlArea2DExt> controlExt;
    std::unique_ptr<PreviewWindows2DExt> previewWindows2DExt;
    std::unique_ptr<Preview3DExt> preview3DExt;
    std::unique_ptr<SensorConfigurationExt> sensorConfigurationExt;

// Create global object for convenience in other functions
public:
    void onFinishedRender() override {

    }

    void onDetach() override {

    }

    void onAttach() override {
        controlExt = std::make_unique<ControlArea2DExt>();
        previewWindows2DExt = std::make_unique<PreviewWindows2DExt>();
        preview3DExt = std::make_unique<Preview3DExt>();
        sensorConfigurationExt = std::make_unique<SensorConfigurationExt>();
    }

    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        bool allUnavailable = true;
        for (auto &d: handles->devices) {
            if (d.state == CRL_STATE_ACTIVE)
                allUnavailable = false;
        }

        if (allUnavailable) {
            // Use background color again
            handles->clearColor[0] = VkRender::Colors::CRLCoolGray.x;
            handles->clearColor[1] = VkRender::Colors::CRLCoolGray.y;
            handles->clearColor[2] = VkRender::Colors::CRLCoolGray.z;
            handles->clearColor[3] = VkRender::Colors::CRLCoolGray.w;
            return;
        }

        for (auto &dev: handles->devices) {
            if (dev.state != CRL_STATE_ACTIVE)
                continue;
            // Control page


            // Viewing page
            createViewingArea(handles, dev);

            ImGui::BeginGroup();
            if (dev.selectedPreviewTab == CRL_TAB_2D_PREVIEW || !dev.extend3DArea)
                createControlArea(handles, dev);

            ImGui::EndGroup();


        }
    }

private:


    void createViewingArea(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {

        /** CREATE TOP NAVIGATION BAR **/
        bool pOpen = true;
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse;

        bool is3DAreaExtended = (dev.selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD && dev.extend3DArea);

        ImVec2 windowPos = is3DAreaExtended ?
                           ImVec2(handles->info->sidebarWidth, 0) :
                           ImVec2(handles->info->sidebarWidth + handles->info->controlAreaWidth, 0);

        ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);

        float viewAreaWidth = is3DAreaExtended ?
                              handles->info->width - handles->info->sidebarWidth :
                              handles->info->width - handles->info->sidebarWidth - handles->info->controlAreaWidth;
        handles->info->viewingAreaWidth = viewAreaWidth;

        ImGui::SetNextWindowSize(ImVec2(handles->info->viewingAreaWidth, handles->info->viewingAreaHeight),
                                 ImGuiCond_Always);
        ImGui::Begin("ViewingArea", &pOpen, window_flags);

        ImGui::Dummy(ImVec2((is3DAreaExtended ?
                             (handles->info->viewingAreaWidth) / 2 :
                             (handles->info->viewingAreaWidth / 2)) - 80.0f, 0.0f));
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, dev.selectedPreviewTab == CRL_TAB_2D_PREVIEW ?
                                               VkRender::Colors::CRLRedActive :
                                               VkRender::Colors::CRLRed);

        if (ImGui::Button("2D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = CRL_TAB_2D_PREVIEW;
            Log::Logger::getInstance()->info("Profile {}: 2D preview pressed", dev.name.c_str());
            handles->clearColor[0] = VkRender::Colors::CRLCoolGray.x;
            handles->clearColor[1] = VkRender::Colors::CRLCoolGray.y;
            handles->clearColor[2] = VkRender::Colors::CRLCoolGray.z;
            handles->clearColor[3] = VkRender::Colors::CRLCoolGray.w;
            handles->usageMonitor->userClickAction("2D", "Button", ImGui::GetCurrentWindow()->Name);

        }
        ImGui::PopStyleColor(); // Btn Color

        ImGui::PushStyleColor(ImGuiCol_Button, dev.selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD ?
                                               VkRender::Colors::CRLRedActive :
                                               VkRender::Colors::CRLRed);

        ImGui::SameLine();

        if (ImGui::Button("3D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = CRL_TAB_3D_POINT_CLOUD;
            Log::Logger::getInstance()->info("Profile {}: 3D preview pressed", dev.name.c_str());
            handles->clearColor[0] = VkRender::Colors::CRL3DBackground.x;
            handles->clearColor[1] = VkRender::Colors::CRL3DBackground.y;
            handles->clearColor[2] = VkRender::Colors::CRL3DBackground.z;
            handles->clearColor[3] = VkRender::Colors::CRL3DBackground.w;
            handles->usageMonitor->userClickAction("3D", "Button", ImGui::GetCurrentWindow()->Name);

        }

        ImGui::PopStyleColor(); // Btn Color

        // Extend 3D area and hide control area

        ImGui::PushStyleColor(ImGuiCol_Button, !is3DAreaExtended ?
                                               VkRender::Colors::CRLRedActive :
                                               VkRender::Colors::CRLRed);


        if (dev.selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD) {
            // Maintain position of buttons in viewing bar even if 3D area is extended
            ImGui::SameLine(0.0f, (is3DAreaExtended ?
                                   (handles->info->viewingAreaWidth) / 2 :
                                   (handles->info->viewingAreaWidth / 2)) - 260.0f);
            std::string btnLabel = dev.extend3DArea ? "Show Control Tab" : "Hide Control Tab";
            if (ImGui::Button(btnLabel.c_str(), ImVec2(150.0f, 20.0f))) {
                handles->usageMonitor->userClickAction(btnLabel, "Button", ImGui::GetCurrentWindow()->Name);

                dev.extend3DArea = dev.extend3DArea ? false : true;
            }

        }
        ImGui::PopStyleColor(); // Btn Color


        windowPos.y += ImGui::GetWindowHeight(); // add this window height to windowpos so next window is
        ImGui::End();
        ImGui::PopStyleColor(); // Bg color
        handles->info->viewingAreaWindowPos = windowPos;
        previewWindows2DExt->onUIRender(handles);
    }


    void createControlArea(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(3.0f, 5.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, handles->info->scrollbarSize);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 0.0f);

        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->controlAreaWidth, handles->info->controlAreaHeight - 20.0f));
        ImGui::Begin("ControlArea", &pOpen, window_flags | ImGuiWindowFlags_NoBringToFrontOnFocus);

        /// DRAW EITHER 2D or 3D Control TAB. ALSO TOP TAB BARS FOR CONTROLS OR SENSOR PARAM
        for (auto &d: handles->devices) {
            // Create dropdown
            if (d.state == CRL_STATE_ACTIVE) {

                ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
                ImVec2 framePadding = ImGui::GetStyle().FramePadding;
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 5.0f));
                ImGui::PushFont(handles->info->font15);

                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {

                    /// Calculate spaces for centering tab bar text
                    float tabBarWidth = handles->info->controlAreaWidth / handles->info->numControlTabs;
                    ImGui::SetNextItemWidth(tabBarWidth);
                    std::string tabLabel = "Preview Control";
                    float labelSize = ImGui::CalcTextSize(tabLabel.c_str()).x;
                    float startPos = (tabBarWidth / 2) - (labelSize / 2);
                    float spaceSize = ImGui::CalcTextSize(std::string(" ").c_str()).x;
                    std::string spaces(int(startPos / spaceSize), ' ');

                    if (ImGui::BeginTabItem((spaces + tabLabel).c_str())) {
                        ImGui::PushFont(handles->info->font13);

                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);

                        dev.controlTabActive = CRL_TAB_PREVIEW_CONTROL;
                        if (dev.selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD)
                            preview3DExt->onUIRender(handles);
                        else
                            buildPreviewControlTab(handles, dev);

                        ImGui::PopStyleVar();//Framepadding
                        ImGui::PopFont();
                        ImGui::EndTabItem();
                    }
                    if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                        handles->usageMonitor->userClickAction("Preview Control", "tab",
                                                               ImGui::GetCurrentWindow()->Name);
                    /// Calculate spaces for centering tab bar text
                    ImGui::SetNextItemWidth(tabBarWidth);
                    tabLabel = "Sensor Config";
                    labelSize = ImGui::CalcTextSize(tabLabel.c_str()).x;
                    startPos = (tabBarWidth / 2) - (labelSize / 2);
                    spaces = std::string(int(startPos / spaceSize), ' ');

                    if (ImGui::BeginTabItem((spaces + tabLabel).c_str())) {
                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);
                        ImGui::PushFont(handles->info->font13);

                        dev.controlTabActive = CRL_TAB_SENSOR_CONFIG;
                        sensorConfigurationExt->onUIRender(handles);
                        ImGui::PopStyleVar();//Framepadding
                        ImGui::PopFont();
                        ImGui::EndTabItem();
                    }
                    if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                        handles->usageMonitor->userClickAction("Sensor Config", "tab",
                                                               ImGui::GetCurrentWindow()->Name);

                    ImGui::EndTabBar();
                }
                ImGui::PopFont();

                ImGui::PopStyleVar(); // Framepadding
            }
        }

        // Draw border between control and viewing area
        ImVec2 lineMin(handles->info->sidebarWidth + handles->info->controlAreaWidth - 3.0f, 0.0f);
        ImVec2 lineMax(lineMin.x + 3.0f, handles->info->height);
        ImGui::GetWindowDrawList()->AddRectFilled(lineMin, lineMax, ImColor(VkRender::Colors::CRLGray424Main), 0.0f,
                                                  0);
        ImGui::Dummy(ImVec2(0.0f, handles->info->height - ImGui::GetCursorPosY()));
        ImGui::End();
        ImGui::PopStyleVar(5);
        ImGui::PopStyleColor();

    }

    void buildPreviewControlTab(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        ImVec2 size = ImVec2(65.0f, 50.0f);
        ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
        ImVec2 uv1 = ImVec2(1.0f, 1.0f);

        ImVec4 bg_col = VkRender::Colors::CRLCoolGray;         // Match bg color
        ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint
        ImGui::Dummy(ImVec2(40.0f, 40.0f));

        // Text
        ImGui::Dummy(ImVec2(40.0f, 0.0));
        ImGui::SameLine();
        ImGui::PushFont(handles->info->font18);
        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
        ImGui::Text("Set layout");
        ImGui::PopStyleColor();

        // Image buttons
        ImGui::Dummy(ImVec2(00.0f, 5.0));
        ImGui::Dummy(ImVec2(40.0f, 0.0));
        ImGui::SameLine();
        if (ImGui::ImageButton("Single", handles->info->imageButtonTextureDescriptor[6], size, uv0,
                               uv1,
                               bg_col, tint_col)) {
            Log::Logger::getInstance()->info("Single Layout pressed.");
            dev.layout = CRL_PREVIEW_LAYOUT_SINGLE;
            handles->usageMonitor->userClickAction("Single", "ImageButton", ImGui::GetCurrentWindow()->Name);

        }
        ImGui::SameLine(0, 20.0f);
        if (ImGui::ImageButton("Double", handles->info->imageButtonTextureDescriptor[7], size, uv0,
                               uv1,
                               bg_col, tint_col)) {
            dev.layout = CRL_PREVIEW_LAYOUT_DOUBLE;
            handles->usageMonitor->userClickAction("Double", "ImageButton", ImGui::GetCurrentWindow()->Name);

        }
        ImGui::SameLine(0, 20.0f);
        if (ImGui::ImageButton("Quad", handles->info->imageButtonTextureDescriptor[8], size, uv0,
                               uv1,
                               bg_col, tint_col)) {
            dev.layout = CRL_PREVIEW_LAYOUT_QUAD;
            handles->usageMonitor->userClickAction("Quad", "ImageButton", ImGui::GetCurrentWindow()->Name);
        }

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
        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
        ImGui::Text("Set sensor resolution");
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(00.0f, 7.0));

        for (size_t i = 0; i < dev.channelInfo.size(); ++i) {
            if (dev.channelInfo[i].state != CRL_STATE_ACTIVE)
                continue;

            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();

            // Resolution selection box
            ImGui::SetNextItemWidth(250);
            std::string resLabel = "##Resolution" + std::to_string(i);

            Log::Logger::getInstance()->traceWithFrequency("Display available modes", 60 * 10,
                                                           "Presented modes to user: ");
            for (const auto &src: dev.channelInfo[i].modes) {
                Log::Logger::getInstance()->traceWithFrequency("Tag:" + src, 60 * 10, "{}", src);
            }

            auto &chInfo = dev.channelInfo[i];
            if (chInfo.state != CRL_STATE_ACTIVE)
                continue;
            if (ImGui::BeginCombo(resLabel.c_str(),
                                  Utils::cameraResolutionToString(chInfo.selectedResolutionMode).c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (size_t n = 0; n < chInfo.modes.size(); n++) {
                    const bool is_selected = (chInfo.selectedModeIndex == n);
                    if (ImGui::Selectable(chInfo.modes[n].c_str(), is_selected)) {
                        chInfo.selectedModeIndex = static_cast<uint32_t>(n);
                        chInfo.selectedResolutionMode = Utils::stringToCameraResolution(
                                chInfo.modes[chInfo.selectedModeIndex]);
                        chInfo.updateResolutionMode = true;
                        handles->usageMonitor->userClickAction("Resolution", "combo", ImGui::GetCurrentWindow()->Name);

                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }
        // Mid separator
        ImGui::Dummy(ImVec2(0.0f, 50.0f));
        ImVec2 posMin = ImGui::GetCursorScreenPos();
        ImVec2 posMax = posMin;
        posMax.x += handles->info->controlAreaWidth;
        posMax.y += 2.0f;
        ImGui::GetWindowDrawList()->AddRectFilled(posMin, posMax,
                                                  ImColor(VkRender::Colors::CRLGray421)); // Separator

        ImGui::Separator();

        controlExt->onUIRender(handles);

    }

};

#endif //MULTISENSE_INTERACTIONMENU_H
