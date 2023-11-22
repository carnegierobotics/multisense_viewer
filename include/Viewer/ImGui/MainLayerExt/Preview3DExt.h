//
// Created by magnus on 9/29/23.
//

#ifndef MULTISENSE_VIEWER_PREVIEW3DEXT_H
#define MULTISENSE_VIEWER_PREVIEW3DEXT_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/ImGui/LayerUtils.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

class Preview3DExt : public VkRender::Layer {
private:

    std::future<std::string> savePointCloudFolderFuture;
    std::future<std::string> saveIMUFolderFuture;

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
            buildConfigurationTab3D(handles, dev);

            ImGui::Dummy(ImVec2(0.0f, 5.0));
            ImGui::Separator();
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            // Draw Recording options
            ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray421);
            ImGui::PushStyleColor(ImGuiCol_Header, VkRender::Colors::CRLRedTransparent);
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, VkRender::Colors::CRLRedActive);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, VkRender::Colors::CRLRedHover);
            ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 10.0f);

            createIMUHeader(handles, dev);
            ImGui::Dummy(ImVec2());
            createPointCloudHeader(handles, dev);


            ImGui::PopStyleColor(4);
            ImGui::PopStyleVar();
        }
    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }

    void buildConfigurationTab3D(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        // Section 1. 3D Viewer
        {
            ImGui::Dummy(ImVec2(40.0f, 40.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::PushFont(handles->info->font15);
            ImGui::Text("Set sensor resolution");
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

        // Section 2
        {            // Check if mouse hover a window
            ImGui::Dummy(ImVec2(0.0f, 15.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::PushFont(handles->info->font15);
            ImGui::Text("Camera type");
            ImGui::PopFont();
            ImGui::PopStyleColor();
            ImGui::Dummy(ImVec2(40.0f, 10.0));
            ImGui::Dummy(ImVec2(40.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            handles->camera.reset = false;
            if (ImGui::RadioButton("Arcball", &handles->camera.type, 0)) {
                handles->usageMonitor->userClickAction("Arcball", "RadioButton", ImGui::GetCurrentWindow()->Name);
                handles->camera.reset = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Flycam", &handles->camera.type, 1)) {
                handles->usageMonitor->userClickAction("Flycam", "RadioButton", ImGui::GetCurrentWindow()->Name);
                handles->camera.reset = true;
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
            handles->camera.reset |= ImGui::Button(
                    "Reset camera position"); // OR true due to resetCamera may be set by clicking radio buttons above
            if (handles->camera.reset) {
                handles->usageMonitor->userClickAction("Reset camera position", "Button",
                                                       ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopStyleColor(2);
        }
        /*

        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);

        ImGui::Dummy(ImVec2(0.0f, 5.0));



         */
    }


    void createIMUHeader(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {


        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 1.0f);
        //ImGui::PushFont(handles->info->font15);
        if (ImGui::CollapsingHeader("IMU", 0)) {

            VkRender::LayerUtils::WidgetPosition pos;
            pos.paddingX = 10.0f;
            pos.textColor = VkRender::Colors::CRLTextGray;

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5.0f);
            ImGui::BeginChild("Imu 3D child", ImVec2(0.0f, 200.0f));

            if (!dev.hasImuSensor) {
                // IMU
                ImGui::Dummy(ImVec2(0.0f, 15.0));
                ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
                ImGui::SameLine();

                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::TextRedColor);
                ImGui::Text("IMU sensor were not found on the device");
                ImGui::PopStyleColor(); // ImGuiCol_Text

                ImGui::BeginDisabled();
            }

            // IMU
            ImGui::Dummy(ImVec2(0.0f, 15.0));
            ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            if (ImGui::Checkbox("Enable IMU", &dev.enableIMU)) {
                handles->usageMonitor->userClickAction("Enable IMU", "Checkbox", ImGui::GetCurrentWindow()->Name);
            }

            ImGui::Dummy(ImVec2(0.0f, 5.0));

            VkRender::LayerUtils::createWidgets(handles, WIDGET_PLACEMENT_IMU, pos);

            { // Save point cloud
                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
                ImGui::SameLine();
                ImGui::Text("Save IMU data to file");
                ImGui::PopStyleColor(); // Text Color grey
                ImGui::SameLine();
                ImGui::HelpMarker(
                        "Record the IMU data to file. The gyro data is saved to gyro.txt as (time (s), dx, dy, dz\nThe accelerometer data is saved to accel.txt as (time (s), x, y, z)",
                        VkRender::Colors::CRLTextWhite);
                // if start then show gif spinner

                ImGui::Dummy(ImVec2(0.0f, 3.0));
                ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
                ImGui::SameLine();
                ImVec2 btnSize(70.0f, 20.0f);

                std::string btnText = dev.record.imu ? "Stop" : "Start";
                if (ImGui::Button((btnText + "##imu").c_str(), btnSize) &&
                    dev.record.imuSaveFolder != "/Path/To/Folder/") {
                    dev.record.imu = !dev.record.imu;
                    handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

                }
                ImGui::SameLine();


                if (ImGui::Button("Set Dir##imu", btnSize)) {
                    if (!saveIMUFolderFuture.valid())
                        saveIMUFolderFuture = std::async(VkRender::LayerUtils::selectFolder);
                    handles->usageMonitor->userClickAction("Choose Dir", "Button", ImGui::GetCurrentWindow()->Name);

                }

                if (saveIMUFolderFuture.valid()) {
                    if (saveIMUFolderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        std::string selectedFolder = saveIMUFolderFuture.get(); // This will also make the future invalid
                        if (!selectedFolder.empty()) {
                            // Do something with the selected folder
                            dev.record.imuSaveFolder = selectedFolder;
                        }
                    }
                }

                ImGui::SameLine();
                ImGui::SetNextItemWidth(
                        handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);

                ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

                std::string hint = "/Path/To/Dir";
                ImGui::CustomInputTextWithHint("##SaveFolderLocationIMU", hint.c_str(),
                                               &dev.record.imuSaveFolder,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::PopStyleColor();

                if (!dev.hasImuSensor)
                    ImGui::EndDisabled();

                ImGui::EndChild();

            }
        }
    }

    void createPointCloudHeader(VkRender::GuiObjectHandles *handles, VkRender::Device &dev) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 1.0f);

        if (ImGui::CollapsingHeader("Point Cloud ", 0)) {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5.0f);
            ImGui::BeginChild("Point cloud 3D child", ImVec2(0.0f, 200.0f));

            VkRender::LayerUtils::WidgetPosition pos;
            pos.paddingX = 10.0f;
            pos.textColor = VkRender::Colors::CRLTextGray;

            ImGui::Dummy(ImVec2(0.0f, 15.0));
            ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::Text("Color Source:");
            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
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
            ImGui::Dummy(ImVec2(0.0f, 5.0));

            VkRender::LayerUtils::createWidgets(handles, WIDGET_PLACEMENT_POINTCLOUD, pos);


            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            ImGui::Text("Save current point cloud as .ply file");
            ImGui::PopStyleColor(); // Text Color grey
            ImGui::SameLine();
            ImGui::HelpMarker("Disparity image to point-cloud conversion is performed on the cpu and therefore slow. Be patient, especially for full res images.");
            ImGui::Dummy(ImVec2(0.0f, 3.0));
            ImGui::Dummy(ImVec2(pos.paddingX, 0.0));
            ImGui::SameLine();
            ImVec2 btnSize(70.0f, 20.0f);

            std::string btnText = dev.record.pointCloud ? "Stop" : "Start";
            if (ImGui::Button((btnText + "##pointcloud").c_str(), btnSize) &&
                dev.record.pointCloudSaveFolder != "/Path/To/Folder/") {
                dev.record.pointCloud = !dev.record.pointCloud;
                handles->usageMonitor->userClickAction(btnText, "Button", ImGui::GetCurrentWindow()->Name);

            }
            ImGui::SameLine();

            if (ImGui::Button("Choose Dir##pointcloud", btnSize)) {
                if (!savePointCloudFolderFuture.valid())
                    savePointCloudFolderFuture = std::async(VkRender::LayerUtils::selectFolder);

                handles->usageMonitor->userClickAction("Choose Dir", "Button", ImGui::GetCurrentWindow()->Name);

            }

            if (savePointCloudFolderFuture.valid()) {
                if (savePointCloudFolderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    std::string selectedFolder = savePointCloudFolderFuture.get(); // This will also make the future invalid
                    if (!selectedFolder.empty()) {
                        // Do something with the selected folder
                        dev.record.pointCloudSaveFolder = selectedFolder;
                    }
                }
            }

            ImGui::SameLine();
            ImGui::SetNextItemWidth(
                    handles->info->controlAreaWidth - ImGui::GetCursorPosX() - btnSize.x - 8.0f);

            ImGui::PushStyleColor(ImGuiCol_TextDisabled, VkRender::Colors::CRLTextWhiteDisabled);

            std::string hint = "/Path/To/Dir";
            ImGui::CustomInputTextWithHint("##SaveFolderLocationPointCloud", hint.c_str(),
                                           &dev.record.pointCloudSaveFolder,
                                           ImGuiInputTextFlags_AutoSelectAll);
            ImGui::PopStyleColor();

            ImGui::EndChild();
        }
    }


};

DISABLE_WARNING_POP
#endif //MULTISENSE_VIEWER_PREVIEW3DEXT_H
