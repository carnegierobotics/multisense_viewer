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
#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/LayerUtils.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/**
 * @brief A UI Layer drawn by \refitem GuiManager.
 * To add an additional UI layer see \refitem LayerExample.
 */
class ControlArea2DExt : public VkRender::Layer {
private:
    enum Compression {
        custom_metadata = 0,
        auto_metadata = 1
    };
public:

    std::future<std::string> folderFuture;


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        ImGui::Dummy(ImVec2(0.0f, 30.0f));
        ImVec2 pos = ImGui::GetCursorPos();
        //pos.x += headerPadding;
        ImGui::SetCursorPos(pos);
        //ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 5.0f));

        ImGui::BeginChild("Configuration2DExtension", ImVec2(handles->info->controlAreaWidth -
                                                             handles->info->scrollbarSize,
                                                             0.0f), false);
        for (auto &dev: handles->devices) {
            if (dev.state != VkRender::CRL_STATE_ACTIVE)
                continue;

            createRecordingHeader(handles, dev);

            VkRender::LayerUtils::WidgetPosition widgetOptions;
            widgetOptions.paddingX = 5.0f;
            widgetOptions.maxElementWidth = handles->info->controlAreaWidth - widgetOptions.paddingX;
            widgetOptions.textColor = VkRender::Colors::CRLTextGray;
            VkRender::LayerUtils::createWidgets(handles, WIDGET_PLACEMENT_MULTISENSE_RENDERER, widgetOptions);
        }

        ImGui::EndChild();
        //ImGui::PopStyleVar();

    }


    void createRecordingHeader(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {

        // Draw Recording options
        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray421);
        ImGui::PushStyleColor(ImGuiCol_Header, VkRender::Colors::CRLRedTransparent);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, VkRender::Colors::CRLRedActive);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, VkRender::Colors::CRLRedHover);
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 10.0f);
        float windowPaddingLeft = 25.0f;
        {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 1.0f);
            //ImGui::PushFont(handles->info->font15);
            if (ImGui::CollapsingHeader("Recording", 0)) {
                ImGui::PushFont(handles->info->font13);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5.0f);
                ImGui::BeginChild("Recording_child", ImVec2(0.0f, 175.0f));
                ImGui::Dummy(ImVec2(5.0f, 10.0f));
                ImGui::Dummy(ImVec2(0.0f, 0.0f));
                ImGui::SameLine();

                ImGui::HelpMarker(
                        "Capture frames displayed in the preview windows using the specified format. Note: Due to hardware constraints and the number of active streams, it's possible some frames might not be saved.",
                        VkRender::Colors::CRLTextWhite);
                ImGui::SameLine();
                ImGui::PushFont(handles->info->font13);
                ImGui::Text("Export Video Frames as:");
                ImGui::PopFont();
                ImGui::SameLine();

                ImVec2 btnSize(ImGui::CalcTextSize("Export Video Frames as:").x, 25.0f);

                if (dev.record.frame) {
                    ImGui::BeginDisabled();
                }

                /** @brief Drop down for compression selection */
                {
                    static std::vector<std::string> saveFormat = {"Select format:", "tiff", "png", "rosbag"};
                    static size_t selector = 0;

                    ImGui::SetNextItemWidth(ImGui::GetWindowSize().x - ImGui::GetCursorPosX() - 5.0f);
                    if (ImGui::BeginCombo("##Compression", saveFormat[selector].c_str(),
                                          ImGuiComboFlags_HeightSmall)) {
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
                }

                ImGui::Dummy(ImVec2(0.0f, 0.0f));
                ImGui::Dummy(ImVec2(windowPaddingLeft, 0.0f));
                ImGui::SameLine();
                /// Record Location Button and file dialog
                {
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLDarkGray425);

                    if (ImGui::Button("Record Location", btnSize)) {
                        if (!folderFuture.valid())
                            folderFuture = std::async(VkRender::LayerUtils::selectFolder, "");
                        handles->usageMonitor->userClickAction("Choose Location", "Button",
                                                               ImGui::GetCurrentWindow()->Name);
                    }

                    if (folderFuture.valid()) {
                        if (folderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            std::string selectedFolder = folderFuture.get(); // This will also make the future invalid
                            if (!selectedFolder.empty()) {
                                // Do something with the selected folder
                                dev.record.frameSaveFolder = selectedFolder;
                            }
                        }
                    }


                    ImGui::SameLine();

                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                                        ImVec2(4.0f, (btnSize.y / 2) - (ImGui::CalcTextSize("size").y) + 6));
                    ImGui::SetNextItemWidth(ImGui::GetWindowSize().x - ImGui::GetCursorPosX() - 5.0f);
                    ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);
                    ImGui::CustomInputTextWithHint("##SaveFolderLocation", hint.c_str(), &dev.record.frameSaveFolder,
                                                   ImGuiInputTextFlags_AutoSelectAll);
                    ImGui::PopStyleColor();
                    ImGui::PopStyleVar();

                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));

                }
                /// RADIOBUTTONS
                ImGui::Dummy(ImVec2(0.0f, 0.0f));
                ImGui::Dummy(ImVec2(windowPaddingLeft, 0.0f));
                ImGui::SameLine();

                {

                    if (ImGui::RadioButton("Custom metadata", &dev.record.metadata.custom, 1)) {
                    };
                    ImGui::SameLine(0.0f, 25.0f);
                    if (ImGui::RadioButton("Auto gen. metadata", &dev.record.metadata.custom, 0)) {

                    };
                }


                if (dev.record.metadata.custom) {
                    ImGui::Dummy(ImVec2(0.0f, 0.0f));
                    ImGui::Dummy(ImVec2(windowPaddingLeft, 0.0f));
                    ImGui::SameLine();
                    if (ImGui::Button("Set custom metadata", btnSize)) {
                        dev.record.showCustomMetaDataWindow = true;
                    }
                }

                if (dev.record.frame) {
                    ImGui::EndDisabled();
                }

                // Check if the user has set a filepath
                if (strlen(dev.record.frameSaveFolder.c_str()) <= 0) {
                    ImGui::BeginDisabled();
                }


                ImGui::Dummy(ImVec2(0.0f, 0.0f));
                ImGui::Dummy(ImVec2(windowPaddingLeft, 0.0f));
                ImGui::SameLine();
                std::string btnText = dev.record.frame ? "Stop" : "Start";
                if (ImGui::Button(btnText.c_str(), btnSize) && dev.record.frameSaveFolder != "/Path/To/Folder/") {
                    dev.record.frame = !dev.record.frame;
                    handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

                    // If start write the metadata contents to file and push save calibration task
                    if (btnText == "Start") {
                        dev.record.metadata.JSON.clear();
                        if (dev.record.metadata.custom)
                            Utils::parseCustomMetadataToJSON(&dev);
                        else
                            Utils::parseMetadataToJSON(&dev);

                        dev.parameters.calib.save = true;
                        dev.parameters.calib.saveCalibrationPath = dev.record.frameSaveFolder;
                        writeJSON(dev);
                    }
                }

                // Check if the user has set a filepath
                if (strlen(dev.record.frameSaveFolder.c_str()) > 0 && !dev.record.frame) {
                    if (std::filesystem::exists(dev.record.frameSaveFolder) &&
                        !std::filesystem::is_empty(dev.record.frameSaveFolder)) {
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::TextRedColor);
                        ImGui::Text("Record folder is not empty");
                        ImGui::PopStyleColor();
                    }
                }
                // Check if the user has input something
                if (strlen(dev.record.frameSaveFolder.c_str()) <= 0) {
                    ImGui::EndDisabled();
                }
                ImGui::PopStyleColor();
                ImGui::PopStyleVar();
                ImGui::PopFont();

                ImGui::EndChild();
            }
            //ImGui::PopFont();

            /** Next feature header here... **/
            /*
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 1.0f);
            if (ImGui::CollapsingHeader("Other", 0)){
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5.0f);
                ImGui::BeginChild("Other_child", ImVec2(0.0f, 150.0f));

                ImGui::EndChild();
            }
             */
        }

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar();
    }

    void writeJSON(VkRender::Device &dev) {
        // Write JSON to file (using nlohmann/json)
        if (dev.record.metadata.parsed) {
            std::filesystem::path saveFolder = dev.record.frameSaveFolder;
            saveFolder.append("metadata.json");
            std::ofstream file(saveFolder);
            file << dev.record.metadata.JSON.dump(4);  // 4 spaces as indentation
            Log::Logger::getInstance()->info("Wrote metadata JSON to file {}", saveFolder.string());
        }
    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }

private:
#ifdef WIN32
    std::string hint = "C:\\Path\\To\\Dir";
#else
    std::string hint = "/Path/To/Dir";
#endif
};

DISABLE_WARNING_POP
#endif //ControlAreaExtension
