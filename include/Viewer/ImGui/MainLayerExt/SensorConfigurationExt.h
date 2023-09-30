//
// Created by magnus on 9/29/23.
//

#ifndef MULTISENSE_VIEWER_SENSORCONFIGURATIONEXT_H
#define MULTISENSE_VIEWER_SENSORCONFIGURATIONEXT_H
#include <ImGuiFileDialog.h>

#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

class SensorConfigurationExt   : public VkRender::Layer {
private:
    ImGuiFileDialog chooseIntrinsicsDialog;
    ImGuiFileDialog chooseExtrinsicsDialog;
    ImGuiFileDialog saveCalibrationDialog;
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> showSavedTimer;
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> showSetTimer;
    std::string setCalibrationFeedbackText;

public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles* handles) override {
        for (auto &dev: handles->devices) {
            if (dev.state != CRL_STATE_ACTIVE)
                continue;
            buildConfigurationTab(handles, dev);
        }
    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }

    void buildConfigurationTab(VkRender::GuiObjectHandles *handles, VkRender::Device &d) {
        {
            float textSpacing = 90.0f;
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);

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

};
DISABLE_WARNING_POP

#endif //MULTISENSE_VIEWER_SENSORCONFIGURATIONEXT_H
