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

#include <filesystem>
#include <GLFW/glfw3.h>
#include <ImGuiFileDialog.h>
#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/ImGui/ScriptUIAddons.h"
#include "Viewer/ImGui/ControlAreaExtension.h"
#include "Viewer/ImGui/PreviewWindows2DExt.h"
#include "Viewer/ImGui/Preview3DExt.h"

#ifdef WIN32
#else

#include <unistd.h>

#endif


class Preview2D3D : public VkRender::Layer {
private:
    bool page[CRL_PAGE_TOTAL_PAGES] = {false, false, true};
    bool drawActionPage = false;
    ImGuiFileDialog chooseIntrinsicsDialog;
    ImGuiFileDialog chooseExtrinsicsDialog;
    ImGuiFileDialog saveCalibrationDialog;
    ImGuiFileDialog savePointCloudDialog;
    ImGuiFileDialog saveIMUDataDialog;
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> showSavedTimer;
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> showSetTimer;
    std::string setCalibrationFeedbackText;
    std::unique_ptr<ControlAreaExtension> controlExt;
    std::unique_ptr<PreviewWindows2DExt> previewWindows2DExt;
    std::unique_ptr<Preview3DExt> preview3DExt;

// Create global object for convenience in other functions
public:
    void onFinishedRender() override {

    }

    void onDetach() override {

    }

    void onAttach() override {
        controlExt = std::make_unique<ControlAreaExtension>();
        previewWindows2DExt = std::make_unique<PreviewWindows2DExt>();
        preview3DExt = std::make_unique<Preview3DExt>();
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

        buildConfigurationPreview(handles);
    }

private:


    void buildPreview(VkRender::GuiObjectHandles *handles) {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);


        if (ImGui::Button("Back")) {
            page[CRL_PAGE_PREVIEW_DEVICES] = false;
            drawActionPage = true;
        }


        ImGui::PopStyleColor(); // bg color
        ImGui::End();

    }


    void buildConfigurationPreview(VkRender::GuiObjectHandles *handles) {
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
        if (dev.isRemoteHead) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::TextColorGray);
        }
        if (ImGui::Button("3D", ImVec2(75.0f, 20.0f))) {
            dev.selectedPreviewTab = CRL_TAB_3D_POINT_CLOUD;
            Log::Logger::getInstance()->info("Profile {}: 3D preview pressed", dev.name.c_str());
            handles->clearColor[0] = VkRender::Colors::CRL3DBackground.x;
            handles->clearColor[1] = VkRender::Colors::CRL3DBackground.y;
            handles->clearColor[2] = VkRender::Colors::CRL3DBackground.z;
            handles->clearColor[3] = VkRender::Colors::CRL3DBackground.w;
            handles->usageMonitor->userClickAction("3D", "Button", ImGui::GetCurrentWindow()->Name);

        }
        if (dev.isRemoteHead) {
            ImGui::PopItemFlag();
            ImGui::PopStyleColor();
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
        //TODO add call to draw2D
        previewWindows2DExt->onUIRender(handles);
    }




    void createControlArea(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 15.0f);

        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::SetNextWindowSize(ImVec2(handles->info->controlAreaWidth, handles->info->controlAreaHeight));
        ImGui::Begin("ControlArea", &pOpen, window_flags | ImGuiWindowFlags_NoBringToFrontOnFocus);


        for (auto &d: handles->devices) {
            // Create dropdown
            if (d.state == CRL_STATE_ACTIVE) {

                ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
                if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);
                    if (ImGui::BeginTabItem((std::string("Preview Control")).c_str())) {


                        dev.controlTabActive = CRL_TAB_PREVIEW_CONTROL;
                        if (dev.selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD)
                            buildConfigurationTab3D(handles, dev);
                        else
                            buildPreviewControlTab(handles, dev);
                        ImGui::EndTabItem();
                    }
                    if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                        handles->usageMonitor->userClickAction("Preview Control", "tab",
                                                               ImGui::GetCurrentWindow()->Name);

                    ImGui::SetNextItemWidth(handles->info->controlAreaWidth / handles->info->numControlTabs);

                    if (ImGui::BeginTabItem("Sensor Config")) {

                        dev.controlTabActive = CRL_TAB_SENSOR_CONFIG;
                        buildConfigurationTab(handles, dev);
                        ImGui::EndTabItem();
                    }
                    if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                        handles->usageMonitor->userClickAction("Sensor Config", "tab",
                                                               ImGui::GetCurrentWindow()->Name);
                    ImGui::EndTabBar();
                }
            }
        }
        ImGui::PopStyleColor();

        // Draw border between control and viewing area
        ImVec2 lineMin(handles->info->sidebarWidth + handles->info->controlAreaWidth - 3.0f, 0.0f);
        ImVec2 lineMax(lineMin.x + 3.0f, handles->info->height);
        ImGui::GetWindowDrawList()->AddRectFilled(lineMin, lineMax, ImColor(VkRender::Colors::CRLGray424Main), 0.0f,
                                                  0);
        ImGui::Dummy(ImVec2(0.0f, handles->info->height - ImGui::GetCursorPosY()));
        ImGui::End();
        ImGui::PopStyleVar(3);
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
        ImGui::Text("2. Choose Sensor Resolution");
        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::Dummy(ImVec2(00.0f, 7.0));

        for (size_t i = 0; i < dev.channelInfo.size(); ++i) {
            if (dev.channelInfo[i].state != CRL_STATE_ACTIVE)
                continue;

            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            if (dev.isRemoteHead) {
                std::string descriptionText = "Remote head " + std::to_string(i + 1) + ":";
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                ImGui::Text("%s", descriptionText.c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine();
            }

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
        ImGui::Dummy(ImVec2(0.0f, 30.0f));



        controlExt->onUIRender(handles);


    }

    void buildConfigurationTab(VkRender::GuiObjectHandles *handles, VkRender::Device &d) {
        {
            float textSpacing = 90.0f;
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);

            if (d.isRemoteHead) {
                ImGui::Dummy(ImVec2(0.0f, 10.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                if (ImGui::RadioButton("Head 1", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_0))
                    d.parameters.updateGuiParams = true;
                ImGui::SameLine(0, 10.0f);

                if (ImGui::RadioButton("Head 2", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_1))
                    d.parameters.updateGuiParams = true;
                ImGui::SameLine(0, 10.0f);


                if (ImGui::RadioButton("Head 3", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_2))
                    d.parameters.updateGuiParams = true;
                ImGui::SameLine(0, 10.0f);
                if (ImGui::RadioButton("Head 4", reinterpret_cast<int *>(&d.configRemoteHead),
                                       crl::multisense::Remote_Head_3))
                    d.parameters.updateGuiParams = true;
            }
            // Exposure Tab
            ImGui::PushFont(handles->info->font18);
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::Dummy(ImVec2(10.0f, 0.0f));
            ImGui::SameLine();
            ImGui::Text("Stereo camera");
            ImGui::PopFont();

            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLBlueIsh);
            ImGui::SameLine(0, 135.0f);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5.0f);
            ImGui::Text("Hold left ctrl to type in values");
            ImGui::PopStyleColor();

            ImGui::Dummy(ImVec2(0.0f, 15.0f));
            ImGui::Dummy(ImVec2(25.0f, 0.0f));
            ImGui::SameLine();
            std::string txt = "Auto Exp.:";
            ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
            ImGui::Text("%s", txt.c_str());
            ImGui::SameLine(0, textSpacing - txtSize.x);
            if (ImGui::Checkbox("##Enable Auto Exposure", &d.parameters.stereo.ep.autoExposure)) {
                handles->usageMonitor->userClickAction("Enable Auto Exposure", "Checkbox",
                                                       ImGui::GetCurrentWindow()->Name);

            }
            d.parameters.stereo.ep.update = ImGui::IsItemDeactivatedAfterEdit();
            // Draw Manual eposure controls or auto exposure control
            if (!d.parameters.stereo.ep.autoExposure) {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\n Exposure in microseconds \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Exposure:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderInt("##Exposure Value: ", reinterpret_cast<int *>(&d.parameters.stereo.ep.exposure),
                                     20, 30000) && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("Exposure value", "SliderInt",
                                                           ImGui::GetCurrentWindow()->Name);

                }
                d.parameters.stereo.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Gain:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Gain",
                                       &d.parameters.stereo.gain, 1.68f,
                                       14.2f, "%.1f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("Stereo camera Gain", "SliderFloat",
                                                           ImGui::GetCurrentWindow()->Name);

                }
                d.parameters.stereo.update = ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

            } else {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Current Exp:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::Text("%d us", d.parameters.stereo.ep.currentExposure);

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\n Max exposure in microseconds \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Max Exp.:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderInt("##Max",
                                     reinterpret_cast<int *>(&d.parameters.stereo.ep.autoExposureMax), 10,
                                     35000) && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("Max exposure", "SliderInt",
                                                           ImGui::GetCurrentWindow()->Name);

                }
                d.parameters.stereo.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Decay Rate:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderInt("##Decay",
                                     reinterpret_cast<int *>(&d.parameters.stereo.ep.autoExposureDecay), 0, 20) &&
                    ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("Exposure Decay", "SliderInt",
                                                           ImGui::GetCurrentWindow()->Name);

                }
                d.parameters.stereo.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Intensity:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##TargetIntensity",
                                       &d.parameters.stereo.ep.autoExposureTargetIntensity, 0, 1) &&
                    ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("TargetIntensity", "SliderFloat",
                                                           ImGui::GetCurrentWindow()->Name);

                }
                d.parameters.stereo.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Threshold:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Threshold", &d.parameters.stereo.ep.autoExposureThresh,
                                       0, 1) && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("Threshold", "SliderFloat", ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.stereo.ep.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                static char buf1[5] = "0";
                static char buf2[5] = "0";
                static char buf3[5] = "0";
                static char buf4[5] = "0";
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker(
                        "\n Set the Region Of Interest for the auto exposure. Note: by default only the left image is used for auto exposure \n ");
                ImGui::SameLine(0.0f, 5.0f);

                if (ImGui::Button("Set ROI", ImVec2(80.0f, 20.0f))) {
                    handles->usageMonitor->userClickAction("Set ROI", "Button", ImGui::GetCurrentWindow()->Name);

                    try {
                        d.parameters.stereo.ep.autoExposureRoiX = std::stoi(buf1);
                        d.parameters.stereo.ep.autoExposureRoiY = std::stoi(buf2);
                        d.parameters.stereo.ep.autoExposureRoiWidth =
                                std::stoi(buf3) - d.parameters.stereo.ep.autoExposureRoiX;
                        d.parameters.stereo.ep.autoExposureRoiHeight =
                                std::stoi(buf4) - d.parameters.stereo.ep.autoExposureRoiY;
                        d.parameters.stereo.ep.update |= true;
                    } catch (...) {
                        Log::Logger::getInstance()->error(
                                "Failed to parse ROI input. User most likely tried to set empty parameters");
                        d.parameters.stereo.ep.update = false;
                    }
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();
                float posX = ImGui::GetCursorPosX();
                float inputWidth = 15.0f * 2.8;
                ImGui::Text("Upper left corner (x, y)");

                ImGui::SameLine(0, 15.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
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
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            if (ImGui::SliderFloat("##Gamma stereo",
                                   &d.parameters.stereo.gamma, 1.1f,
                                   2.2f, "%.2f") && ImGui::IsItemActivated()) {
                handles->usageMonitor->userClickAction("Gamma stereo", "SliderFloat", ImGui::GetCurrentWindow()->Name);

            }
            // Correct update sequence. This is because gamma and gain was part of general parameters. This will probably be redone in the future once established categories are in place
            if (d.parameters.stereo.ep.autoExposure)
                d.parameters.stereo.update = ImGui::IsItemDeactivatedAfterEdit();
            else
                d.parameters.stereo.update |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::PopStyleColor();
            ImGui::Separator();

            // White Balance
            if (d.hasColorCamera) {
                ImGui::PushFont(handles->info->font18);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(10.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("Aux camera");
                ImGui::PopFont();

                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                std::string txt = "Auto Exp.:";
                ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                if (ImGui::Checkbox("##Enable AUX Auto Exposure", &d.parameters.aux.ep.autoExposure)) {
                    handles->usageMonitor->userClickAction("Aux Auto Exposure", "Checkbox",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.aux.update = ImGui::IsItemDeactivatedAfterEdit();
                // Draw Manual eposure controls or auto exposure control
                if (!d.parameters.aux.ep.autoExposure) {
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(3.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::HelpMarker("\n Exposure in microseconds \n ");
                    ImGui::PopStyleColor();
                    ImGui::SameLine(0.0f, 5);
                    txt = "Exposure:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderInt("##Exposure Value aux: ",
                                         reinterpret_cast<int *>(&d.parameters.aux.ep.exposure),
                                         20, 30000) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("Aux Exposure value", "SliderInt",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Gain:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat("##Gain aux",
                                           &d.parameters.aux.gain, 1.68f,
                                           14.2f, "%.1f") && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("gain aux", "SliderFloat",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                } else {
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Current Exp:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::Text("%d us", d.parameters.aux.ep.currentExposure);

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(3.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::HelpMarker("\n Max exposure in microseconds \n ");
                    ImGui::PopStyleColor();
                    ImGui::SameLine(0.0f, 5);
                    txt = "Max Exp.:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderInt("##MaxAux",
                                         reinterpret_cast<int *>(&d.parameters.aux.ep.autoExposureMax), 10,
                                         35000) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("Max Aux Exposure value", "SliderInt",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Decay Rate:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderInt("##DecayAux",
                                         reinterpret_cast<int *>(&d.parameters.aux.ep.autoExposureDecay), 0, 20) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("DecayAux", "SliderInt",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Intensity:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat("##TargetIntensityAux",
                                           &d.parameters.aux.ep.autoExposureTargetIntensity, 0, 1) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("TargetIntensityAux", "SliderInt",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Threshold:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat("##ThresholdAux", &d.parameters.aux.ep.autoExposureThresh,
                                           0, 1) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("ThresholdAux", "SliderInt",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(3.0f, 0.0f));
                    ImGui::SameLine();
                    static char buf1[5] = "0";
                    static char buf2[5] = "0";
                    static char buf3[5] = "0";
                    static char buf4[5] = "0";
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::HelpMarker(
                            "\n Set the Region Of Interest for the auto exposure. Note: by default only the left image is used for auto exposure \n ");
                    ImGui::SameLine(0.0f, 5.0f);

                    if (ImGui::Button("Set ROI##aux", ImVec2(80.0f, 20.0f))) {

                        handles->usageMonitor->userClickAction("Set ROI##aux", "Button",
                                                               ImGui::GetCurrentWindow()->Name);

                        try {
                            d.parameters.aux.ep.autoExposureRoiX = std::stoi(buf1);
                            d.parameters.aux.ep.autoExposureRoiY = std::stoi(buf2);
                            d.parameters.aux.ep.autoExposureRoiWidth =
                                    std::stoi(buf3) - d.parameters.aux.ep.autoExposureRoiX;
                            d.parameters.aux.ep.autoExposureRoiHeight =
                                    std::stoi(buf4) - d.parameters.aux.ep.autoExposureRoiY;
                            d.parameters.aux.update |= true;
                        } catch (...) {
                            Log::Logger::getInstance()->error(
                                    "Failed to parse ROI input. User most likely tried to set empty parameters");
                            d.parameters.aux.update = false;
                        }
                    }
                    ImGui::PopStyleColor();

                    ImGui::SameLine();
                    float posX = ImGui::GetCursorPosX();
                    float inputWidth = 15.0f * 2.8;
                    ImGui::Text("Upper left corner (x, y)");

                    ImGui::SameLine(0, 15.0f);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(inputWidth);
                    ImGui::InputText("##decimalminXAux", buf1, 5, ImGuiInputTextFlags_CharsDecimal);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(inputWidth);
                    ImGui::InputText("##decimalminYAux", buf2, 5, ImGuiInputTextFlags_CharsDecimal);
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::SetCursorPosX(posX);
                    ImGui::Text("Lower right corner (x, y)");
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(inputWidth);
                    ImGui::InputText("##decimalmaxXAux", buf3, 5, ImGuiInputTextFlags_CharsDecimal);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(inputWidth);
                    ImGui::InputText("##decimalmaxYAux", buf4, 5, ImGuiInputTextFlags_CharsDecimal);
                    ImGui::PopStyleColor();

                }

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Gamma:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Gamma aux",
                                       &d.parameters.aux.gamma, 1.1f,
                                       2.2f, "%.2f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("##Gamma aux", "SliderFloat",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();

                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Auto WB:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                if (ImGui::Checkbox("##Enable AUX auto white balance", &d.parameters.aux.whiteBalanceAuto)) {
                    handles->usageMonitor->userClickAction("##Enable AUX auto white balance", "Checkbox",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));

                if (!d.parameters.aux.whiteBalanceAuto) {
                    ImGui::SameLine();
                    txt = "Red Balance:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat("##WBRed",
                                           &d.parameters.aux.whiteBalanceRed, 0.25f,
                                           4.0f) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("##WBRed", "SliderFloat",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Blue Balance:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat("##WBBlue",
                                           &d.parameters.aux.whiteBalanceBlue, 0.25f,
                                           4.0f) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("##WBBlue", "SliderFloat",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();
                } else {
                    ImGui::SameLine();
                    txt = "Threshold:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat("##WBTreshold",
                                           &d.parameters.aux.whiteBalanceThreshold, 0.0,
                                           1.0f) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("##WBTreshold", "SliderFloat",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Decay Rate:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderInt("##DecayRateWB",
                                         reinterpret_cast<int *>(&d.parameters.aux.whiteBalanceDecay), 0,
                                         20) && ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction("##DecayRateWB", "SliderInt",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                    ImGui::PopStyleColor();
                }

                // Aux sharpening
                {
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(25.0f, 0.0f));
                    ImGui::SameLine();
                    txt = "Sharpening:";
                    txtSize = ImGui::CalcTextSize(txt.c_str());
                    ImGui::Text("%s", txt.c_str());
                    ImGui::SameLine(0, textSpacing - txtSize.x);
                    if (ImGui::Checkbox("##Enable AUX sharpening", &d.parameters.aux.sharpening)) {
                        handles->usageMonitor->userClickAction("##Enable AUX sharpening", "Checkbox",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();

                    if (d.parameters.aux.sharpening) {
                        ImGui::Dummy(ImVec2(0.0f, 5.0f));
                        ImGui::Dummy(ImVec2(25.0f, 0.0f));
                        ImGui::SameLine();
                        txt = "Percentage:";
                        txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x);
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                        if (ImGui::SliderFloat("##sharpeningPercentage",
                                               &d.parameters.aux.sharpeningPercentage, 0.0,
                                               100.0f) && ImGui::IsItemActivated()) {
                            handles->usageMonitor->userClickAction("##sharpeningPercentage", "SliderFloat",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }
                        d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                        ImGui::PopStyleColor();

                        ImGui::Dummy(ImVec2(0.0f, 5.0f));
                        ImGui::Dummy(ImVec2(25.0f, 0.0f));
                        ImGui::SameLine();
                        txt = "Limit:";
                        txtSize = ImGui::CalcTextSize(txt.c_str());
                        ImGui::Text("%s", txt.c_str());
                        ImGui::SameLine(0, textSpacing - txtSize.x);
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                        if (ImGui::SliderInt("##sharpeningLimit",
                                             reinterpret_cast<int *>(&d.parameters.aux.sharpeningLimit), 0,
                                             100) && ImGui::IsItemActivated()) {
                            handles->usageMonitor->userClickAction("##sharpeningLimit", "SliderInt",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }
                        d.parameters.aux.update |= ImGui::IsItemDeactivatedAfterEdit();
                        ImGui::PopStyleColor();
                    }
                }
                ImGui::Separator();
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
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker(
                        "\n If enabled then LEDs are only on when the image sensor is exposing. This significantly reduces the sensor's power consumption \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                std::string txtEnableFlash = "Flash LED:";
                ImVec2 txtSizeEnableFlash = ImGui::CalcTextSize(txtEnableFlash.c_str());
                ImGui::Text("%s", txtEnableFlash.c_str());
                ImGui::SameLine(0, textSpacing - txtSizeEnableFlash.x);
                if (ImGui::Checkbox("##Enable Lights", &d.parameters.light.flashing)) {
                    handles->usageMonitor->userClickAction("##Enable Lights", "Checkbox",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.light.update = ImGui::IsItemDeactivatedAfterEdit();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Duty Cycle :";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Duty_Cycle",
                                       &d.parameters.light.dutyCycle, 0,
                                       100,
                                       "%.0f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("##Duty_Cycle", "SliderFloat",
                                                           ImGui::GetCurrentWindow()->Name);
                } // showing 0 float precision not using int cause underlying libmultisense is a float
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::SliderInt("##LightSelection",
                                 reinterpret_cast<int *>(&d.parameters.light.selection), -1,
                                 3);
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
*/
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\n Light pulses per exposure \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Light Pulses:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Pulses",
                                       reinterpret_cast<float *>(&d.parameters.light.numLightPulses), 0,
                                       60, "%.1f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("##Pulses", "SliderFloat", ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(3.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\n LED startup time in milliseconds \n ");
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, 5);
                txt = "Startup Time:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::SetNextItemWidth(handles->info->controlAreaWidth - 72.0f - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Startup Time",
                                       reinterpret_cast<float *>(&d.parameters.light.startupTime), 0,
                                       60, "%.1f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("##Startup Time", "SliderFloat",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.light.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }
            ImGui::Separator();
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Framerate",
                                       &d.parameters.stereo.fps, 1,
                                       30, "%.1f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("##Framerate", "SliderFloat",
                                                           ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.stereo.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(25.0f, 0.0f));
                ImGui::SameLine();
                txt = "Stereo Filter:";
                txtSize = ImGui::CalcTextSize(txt.c_str());
                ImGui::Text("%s", txt.c_str());
                ImGui::SameLine(0, textSpacing - txtSize.x);
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                if (ImGui::SliderFloat("##Stereo",
                                       &d.parameters.stereo.stereoPostFilterStrength, 0.0f,
                                       1.0f, "%.1f") && ImGui::IsItemActivated()) {
                    handles->usageMonitor->userClickAction("##Stereo", "SliderFloat", ImGui::GetCurrentWindow()->Name);
                }
                d.parameters.stereo.update |= ImGui::IsItemDeactivatedAfterEdit();
                ImGui::PopStyleColor();
            }

            ImGui::Separator();

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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImVec2 btnSize(100.0f, 20.0f);
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - (btnSize.x) - 35.0f);
                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

#ifdef WIN32
                std::string hint = "C:\\Path\\To\\dir";
#else
                std::string hint = "/Path/To/dir";
#endif
                ImGui::CustomInputTextWithHint("##SaveLocation", hint.c_str(), &d.parameters.calib.saveCalibrationPath,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();

                ImGui::SameLine();


                if (ImGui::Button("Choose Dir", btnSize)) {
                    saveCalibrationDialog.OpenDialog("ChooseDirDlgKey", "Choose save location", nullptr,
                                                     ".");
                    handles->usageMonitor->userClickAction("Choose Dir", "Button", ImGui::GetCurrentWindow()->Name);

                }
                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);

                if (d.parameters.calib.saveCalibrationPath == "Path/To/Dir") {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::TextColorGray);
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::Colors::TextColorGray);
                }
                d.parameters.calib.save = ImGui::Button("Get Current Calibration");

                if (d.parameters.calib.saveCalibrationPath == "Path/To/Dir") {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleColor(2);
                }
                ImGui::PopStyleColor();

                if (d.parameters.calib.save) {
                    showSavedTimer = std::chrono::steady_clock::now();
                    handles->usageMonitor->userClickAction("Get Current Calibration", "Button",
                                                           ImGui::GetCurrentWindow()->Name);

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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - (btnSize.x) - 35.0f);
                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

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


                if (ImGui::Button("Choose File##1", btnSize)) {
                    chooseIntrinsicsDialog.OpenDialog("ChooseFileDlgKey", "Choose intrinsics .yml file", ".yml",
                                                      ".");
                    handles->usageMonitor->userClickAction("Choose File##1", "Button", ImGui::GetCurrentWindow()->Name);

                }
                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - (btnSize.x) - 35.0f);
                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

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
                if (ImGui::Button("Choose File##2", btnSize)) {
                    chooseExtrinsicsDialog.OpenDialog("ChooseFileDlgKey", "Choose extrinsics .yml file", ".yml",
                                                      ".");
                    handles->usageMonitor->userClickAction("Choose File##2", "Button", ImGui::GetCurrentWindow()->Name);

                }
                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
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
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);

                if (d.parameters.calib.intrinsicsFilePath == "Path/To/Intrinsics.yml" ||
                    d.parameters.calib.extrinsicsFilePath == "Path/To/Extrinsics.yml") {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::TextColorGray);
                    ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::Colors::TextColorGray);
                }

                if (ImGui::Button("Set New Calibration")) {
                    handles->usageMonitor->userClickAction("Set New Calibration", "Button",
                                                           ImGui::GetCurrentWindow()->Name);
                    // Check if file exist before opening popup
                    bool extrinsicsExists = std::filesystem::exists(d.parameters.calib.extrinsicsFilePath);
                    bool intrinsicsExists = std::filesystem::exists(d.parameters.calib.intrinsicsFilePath);
                    if (extrinsicsExists && intrinsicsExists) {
                        ImGui::OpenPopup("Overwrite calibration?");
                    } else {
                        showSetTimer = std::chrono::steady_clock::now();
                        setCalibrationFeedbackText = "Path(s) not valid";
                    }
                }

                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5.0f, 5.0f));
                if (ImGui::BeginPopupModal("Overwrite calibration?", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                    ImGui::Text(
                            " Setting a new calibration will overwrite the current setting. \n This operation cannot be undone! \n Remember to backup calibration as to not loose the factory calibration \n");
                    ImGui::Separator();

                    if (ImGui::Button("OK", ImVec2(120, 0))) {
                        handles->usageMonitor->userClickAction("OK", "Button", ImGui::GetCurrentWindow()->Name);
                        d.parameters.calib.update = true;
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::SetItemDefaultFocus();
                    ImGui::SameLine();


                    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - ImGui::GetCursorPosX() + 8.0f);
                    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                        handles->usageMonitor->userClickAction("Cancel", "Button", ImGui::GetCurrentWindow()->Name);
                        ImGui::CloseCurrentPopup();
                    }
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

                if (!d.parameters.calib.updateFailed && d.parameters.calib.update) {
                    setCalibrationFeedbackText = "Set calibration. Please reboot camera";
                } else if (d.parameters.calib.updateFailed && d.parameters.calib.update) {
                    setCalibrationFeedbackText = "Failed to set calibration...";
                }

                time = std::chrono::steady_clock::now();
                threeSeconds = 3.0f;
                time_span = std::chrono::duration_cast<std::chrono::duration<float>>(time - showSetTimer);
                if (time_span.count() < threeSeconds) {
                    ImGui::SameLine();
                    ImGui::Text("%s", setCalibrationFeedbackText.c_str());
                }
            }
            ImGui::PopStyleColor();
        }
    }

    void buildConfigurationTab3D(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        // Section 1. 3D Viewer
        {
            ImGui::Dummy(ImVec2(40.0f, 40.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::PushFont(handles->info->font15);
            ImGui::Text("1. Sensor Resolution");
            ImGui::PopFont();
            ImGui::PopStyleColor();
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::SetNextItemWidth(200);
            std::string resLabel = "##Resolution";
            auto &chInfo = dev.channelInfo.front();
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
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::Separator();

        // Section 2
        {            // Check if mouse hover a window
            ImGui::Dummy(ImVec2(0.0f, 15.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::PushFont(handles->info->font15);
            ImGui::Text("2. Camera Type");
            ImGui::PopFont();
            ImGui::PopStyleColor();
            ImGui::Dummy(ImVec2(40.0f, 10.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            dev.resetCamera = false;
            if (ImGui::RadioButton("Arcball", &dev.cameraType, 0)) {
                handles->usageMonitor->userClickAction("Arcball", "RadioButton", ImGui::GetCurrentWindow()->Name);
                dev.resetCamera = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Flycam", &dev.cameraType, 1)) {
                handles->usageMonitor->userClickAction("Flycam", "RadioButton", ImGui::GetCurrentWindow()->Name);
                dev.resetCamera = true;
            }
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));
            ImGui::HelpMarker(
                    "Select between arcball or flycam type. Flycam uses Arrow/WASD keys to move camera and mouse + click to rotate");
            ImGui::PopStyleVar();
            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            dev.resetCamera |= ImGui::Button(
                    "Reset camera position"); // OR true due to resetCamera may be set by clicking radio buttons above
            if (dev.resetCamera) {
                handles->usageMonitor->userClickAction("Reset camera position", "Button",
                                                       ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::Dummy(ImVec2(0.0f, 5.0f));

        ImGui::Separator();

        // Section 3
        {
            ImGui::Dummy(ImVec2(0.0f, 15.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::PushFont(handles->info->font15);
            ImGui::Text("3. Options");
            ImGui::PopStyleColor();
            ImGui::PopFont();

            // IMU
            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            if (ImGui::Checkbox("Enable IMU", &dev.useIMU)) {
                handles->usageMonitor->userClickAction("Enable IMU", "Checkbox", ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopStyleColor();

            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::Text("Color:");
            ImGui::Dummy(ImVec2(40.0f, 3.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            if (ImGui::RadioButton("Grayscale", &dev.useAuxForPointCloudColor, 0)) {
                handles->usageMonitor->userClickAction("Grayscale", "RadioButton", ImGui::GetCurrentWindow()->Name);
            }
            if (!dev.hasColorCamera)
                ImGui::BeginDisabled();
            ImGui::SameLine();
            if (ImGui::RadioButton("Color", &dev.useAuxForPointCloudColor, 1)) {
                handles->usageMonitor->userClickAction("Color", "RadioButton", ImGui::GetCurrentWindow()->Name);
            }
            if (!dev.hasColorCamera) {
                ImGui::SameLine();
                ImGui::EndDisabled();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker("\nColor source is only available if a color imager is present\n\n");
                ImGui::PopStyleColor(); // text color

            }

            ImGui::PopStyleColor();
        }

        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);

        ImGui::Dummy(ImVec2(0.0f, 5.0));

        for (const auto &elem: Widgets::make()->elements) {
            // for each element type
            ImGui::Dummy(ImVec2(0.0f, 0.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            switch (elem.type) {
                case WIDGET_CHECKBOX:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::Checkbox(elem.label, elem.checkbox) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_CHECKBOX",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;

                case WIDGET_FLOAT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat(elem.label, elem.value, elem.minValue, elem.maxValue) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_FLOAT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_INT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderInt(elem.label, elem.intValue, elem.intMin, elem.intMax) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_INT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_TEXT:
                    ImGui::Text("%s", elem.label);
                    break;
            }

        }

        ImGui::PopStyleColor(); // ImGuiCol_Text
        ImGui::Dummy(ImVec2(

                0.0f, 5.0f));

        ImGui::Separator();
        // Section 4
        {
            ImGui::Dummy(ImVec2(0.0f, 15.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::PushFont(handles->info->font15);
            ImGui::Text("4. Recording");
            ImGui::PopFont();

            { // Save point cloud
                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImGui::Text("Save IMU data to file");
                ImGui::PopStyleColor(); // Text Color grey
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                ImGui::HelpMarker(
                        "Record the IMU data to file. The gyro data is saved to gyro.txt as (time (s), dx, dy, dz\nThe accelerometer data is saved to accel.txt as (time (s), x, y, z)");
                // if start then show gif spinner
                ImGui::PopStyleColor();

                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImVec2 btnSize(70.0f, 30.0f);

                std::string btnText = dev.isRecordingIMUdata ? "Stop" : "Start";
                if (ImGui::Button((btnText + "##imu").c_str(), btnSize) &&
                    dev.outputSaveFolderIMUData != "/Path/To/Folder/") {
                    dev.isRecordingIMUdata = !dev.isRecordingIMUdata;
                    handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

                }
                ImGui::SameLine();

                if (ImGui::Button("Choose Dir##imu", btnSize)) {
                    saveIMUDataDialog.OpenDialog("ChooseDirDlgKey", "Choose a Directory", nullptr,
                                                 ".");
                    handles->usageMonitor->userClickAction("Choose Dir", "Button", ImGui::GetCurrentWindow()->Name);

                }

                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                if (saveIMUDataDialog.Display("ChooseDirDlgKey", 0, ImVec2(600.0f, 400.0f),
                                              ImVec2(1200.0f, 1000.0f))) {
                    // action if OK
                    if (saveIMUDataDialog.IsOk()) {
                        std::string filePathName = saveIMUDataDialog.GetFilePathName();
                        dev.outputSaveFolderIMUData = filePathName;
                        // action
                    }
                    // close
                    saveIMUDataDialog.Close();
                }
                ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
                ImGui::PopStyleColor(); // ImGuiCol_WindowBg

                ImGui::SameLine();
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 9.0f));
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);

                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

                std::string hint = "/Path/To/Dir";
                ImGui::CustomInputTextWithHint("##SaveFolderLocationIMU", hint.c_str(),
                                               &dev.outputSaveFolderIMUData,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
            }


            { // Save point cloud
                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                ImGui::Text("Save Point cloud as .ply file");
                ImGui::PopStyleColor(); // Text Color grey

                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(40.0f, 0.0));
                ImGui::SameLine();
                ImVec2 btnSize(70.0f, 30.0f);

                std::string btnText = dev.isRecordingPointCloud ? "Stop" : "Start";
                if (ImGui::Button((btnText + "##pointcloud").c_str(), btnSize) &&
                    dev.outputSaveFolderPointCloud != "/Path/To/Folder/") {
                    dev.isRecordingPointCloud = !dev.isRecordingPointCloud;
                    handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

                }
                ImGui::SameLine();

                if (ImGui::Button("Choose Dir##pointcloud", btnSize)) {
                    savePointCloudDialog.OpenDialog("ChooseDirDlgKey", "Choose a Directory", nullptr,
                                                    ".");
                    handles->usageMonitor->userClickAction("Choose Dir", "Button", ImGui::GetCurrentWindow()->Name);

                }

                // display
                ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                if (savePointCloudDialog.Display("ChooseDirDlgKey", 0, ImVec2(600.0f, 400.0f),
                                                 ImVec2(1200.0f, 1000.0f))) {
                    // action if OK
                    if (savePointCloudDialog.IsOk()) {
                        std::string filePathName = savePointCloudDialog.GetFilePathName();
                        dev.outputSaveFolderPointCloud = filePathName;
                        // action
                    }
                    // close
                    savePointCloudDialog.Close();
                }
                ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
                ImGui::PopStyleColor(); // ImGuiCol_WindowBg

                ImGui::SameLine();
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 9.0f));
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);

                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

                std::string hint = "/Path/To/Dir";
                ImGui::CustomInputTextWithHint("##SaveFolderLocationPointCloud", hint.c_str(),
                                               &dev.outputSaveFolderPointCloud,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
            }
        }

    }

};

#endif //MULTISENSE_INTERACTIONMENU_H
