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

#ifndef MULTISENSE_VIEWER_CONTROL_AREA_EXTENSION_H
#define MULTISENSE_VIEWER_CONTROL_AREA_EXTENSION_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/**
 * @brief A UI Layer drawn by \refitem GuiManager.
 * To add an additional UI layer see \refitem LayerExample.
 */
class ControlAreaExtension : public VkRender::Layer {

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


            // Draw Recording options
            {
                if (ImGui::CollapsingHeader("Recording", 0)) {
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray421);
                    ImGui::BeginChild("Recording_child");
                    ImGui::PopStyleColor();
                    ImGui::Dummy(ImVec2(40.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushFont(handles->info->font18);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                    ImGui::Text("Recording");
                    ImGui::PopFont();
                    ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::HelpMarker(
                            " \n Saves the frames shown in the viewing are to the right to files.  \n Each type of stream is saved in separate folders \n Depending on hardware, active streams, and if you chose \n a compressed method (png)    \n you may not be able to save all frames \n\n Color images are saved as either ppm/png files   ");
                    // if start then show gif spinner
                    ImGui::PopStyleColor();

                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                    ImGui::Dummy(ImVec2(40.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::Text("Save active streams as images to file");
                    ImGui::Spacing();
                    ImGui::PopStyleColor();

                    ImGui::Dummy(ImVec2(40.0f, 0.0f));
                    ImGui::SameLine();
                    ImVec2 btnSize(120.0f, 30.0f);
                    std::string btnText = dev.isRecording ? "Stop" : "Start";
                    if (ImGui::Button(btnText.c_str(), btnSize) && dev.outputSaveFolder != "/Path/To/Folder/") {
                        dev.isRecording = !dev.isRecording;
                        handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

                    }
                    ImGui::SameLine();

                    static std::vector<std::string> saveFormat = {"Select format:", "tiff", "png"};
                    static size_t selector = 0;

                    ImGui::SetNextItemWidth(
                            handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);
                    if (ImGui::BeginCombo("##Compression", saveFormat[selector].c_str(), ImGuiComboFlags_HeightSmall)) {
                        for (size_t n = 0; n < saveFormat.size(); n++) {
                            const bool is_selected = (selector == n);
                            if (ImGui::Selectable(saveFormat[n].c_str(), is_selected)) {
                                selector = n;
                                dev.saveImageCompressionMethod = saveFormat[selector];
                                handles->usageMonitor->userClickAction("Compression", "combo",
                                                                       ImGui::GetCurrentWindow()->Name);

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
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::TextColorGray);
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::Colors::TextColorGray);

                    }
                    {
                        if (ImGui::Button("Choose Location", btnSize)) {
                            ImGuiFileDialog::Instance()->OpenDialog("ChooseDirDlgKey", "Choose a Directory", nullptr,
                                                                    ".");
                            handles->usageMonitor->userClickAction("Choose Location", "Button",
                                                                   ImGui::GetCurrentWindow()->Name);

                        }
                    }

                    ImGui::SameLine();
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 9.0f));
                    ImGui::SetNextItemWidth(
                            handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);

                    ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);
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
                    ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
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

                    ImGui::EndChild();
                }
            }

            bool anyStreamActive = false;
            for (const auto &ch: dev.channelInfo) {
                if (!ch.requestedStreams.empty())
                    anyStreamActive = true;
            }
            if (anyStreamActive) {
                ImGui::Dummy(ImVec2(40.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
                ImGui::Text("Currently Active Streams:");
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextLightGray);
                for (const auto &ch: dev.channelInfo) {
                    if (dev.isRemoteHead) {
                        for (const auto &src: ch.requestedStreams) {
                            ImGui::Dummy(ImVec2(60.0f, 0.0f));
                            ImGui::SameLine();
                            ImGui::Text("Head: %d, Source: %s", ch.index + 1, src.c_str());
                        }
                    } else {
                        for (const auto &src: ch.requestedStreams) {
                            if (src == "Idle")
                                continue;

                            ImGui::Dummy(ImVec2(60.0f, 0.0f));
                            ImGui::SameLine();
                            ImGui::Text("%s", src.c_str());
                        }
                    }
                }
                ImGui::PopStyleColor(2);
            }
        }

    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }
};


#endif //ControlAreaExtension
