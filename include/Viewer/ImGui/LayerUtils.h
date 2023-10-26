//
// Created by mgjer on 20/10/2023.
//

#ifndef MULTISENSE_VIEWER_LAYERUTILS_H
#define MULTISENSE_VIEWER_LAYERUTILS_H

#include <imgui.h>

#include "Viewer/ImGui/Widgets.h"

namespace VkRender::LayerUtils {

    struct WidgetPosition {
        float paddingX = -1.0f;
        ImVec4 textColor = VkRender::Colors::CRLTextWhite;
        bool sameLine = false;

        float maxElementWidth = 300.0f;
    };

    static inline void
    createWidgets(VkRender::GuiObjectHandles *handles, const std::string &area, WidgetPosition pos = WidgetPosition()) {
        for (auto &elem: Widgets::make()->elements[area]) {

            if (pos.paddingX != -1) {
                ImGui::Dummy(ImVec2(pos.paddingX, 0.0f));
                ImGui::SameLine();
            }

            switch (elem.type) {
                case WIDGET_CHECKBOX:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::Checkbox(elem.label.c_str(), elem.checkbox) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_CHECKBOX",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;

                case WIDGET_FLOAT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(pos.maxElementWidth);
                    if (ImGui::SliderFloat(elem.label.c_str(), elem.value, elem.minValue, elem.maxValue) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_FLOAT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_INT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(pos.maxElementWidth);
                    *elem.active = false;
                    ImGui::SliderInt(elem.label.c_str(), elem.intValue, elem.intMin, elem.intMax);
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_INT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                        *elem.active = true;
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_TEXT:
                    ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                    ImGui::Text("%s", elem.label.c_str());
                    ImGui::PopStyleColor();

                    break;
                case WIDGET_INPUT_TEXT:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(pos.maxElementWidth);
                    ImGui::InputText(elem.label.c_str(), elem.buf, 1024, 0);
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_BUTTON:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    *elem.button = ImGui::Button(elem.label.c_str());
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_SELECT_DIR_DIALOG:
                    ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose a file",
                                                            ".*", "/home/magnus/crl/disparity_quality/processed_data/");
                    // If open dialog
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    *elem.button |= ImGui::Button(elem.label.c_str());
                    ImGui::PopStyleColor();
                    if (*elem.button) {
                        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
                        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey", 0, ImVec2(600.0f, 400.0f),
                                                                 ImVec2(1200.0f, 1000.0f))) {
                            // action if OK
                            if (ImGuiFileDialog::Instance()->IsOk()) {
                                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                                // Check if the selected item is a file and not a directory
                                if (std::filesystem::is_regular_file(filePathName)) {
                                    strcpy(elem.buf, filePathName.c_str());
                                    *elem.button = false;
                                } else {
                                    // Optionally show an error message or re-prompt the user
                                }
                            }
                        }
                        ImGui::PopStyleColor();
                        ImGui::PopStyleVar();

                    }
                default:
                    break;
            }
            if (pos.sameLine)
                ImGui::SameLine();
        }
        ImGui::Dummy(ImVec2());
    }
}


#endif //MULTISENSE_VIEWER_LAYERUTILS_H
