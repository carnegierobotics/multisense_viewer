//
// Created by magnus on 9/30/23.
//

#ifndef MULTISENSE_VIEWER_CUSTOMMETADATA_H
#define MULTISENSE_VIEWER_CUSTOMMETADATA_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
class CustomMetadata : public VkRender::Layer {
private:
    ImGuiTextBuffer Buf;
    ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.

public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handle) override {
        for (auto &dev: handle->devices) {
            if (dev.state != CRL_STATE_ACTIVE)
                continue;

            if (dev.record.showCustomMetaDataWindow) {
                ImGui::OpenPopup("set_custom_metadata_modal");
                dev.record.showCustomMetaDataWindow = false;
            }

            showPopup(handle, &dev);
        }
    }

    void showPopup(VkRender::GuiObjectHandles *handles, VkRender::Device *dev) {
        ImGui::SetNextWindowSize(ImVec2(handles->info->metadataWidth, handles->info->metadataHeight), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextLightGray);

        ImGui::PushFont(handles->info->font15);
        if (ImGui::BeginPopupModal("set_custom_metadata_modal", nullptr,
                                   ImGuiWindowFlags_NoDecoration)) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            float width = handles->info->metadataWidth;
            float centerWidth = handles->info->metadataWidth / 2.0f;
            float textPadding = 50.0f;
            float textInputPadding = centerWidth - textPadding;
            /** HEADER FIELD */
            ImVec2 popupDrawPos = ImGui::GetCursorScreenPos();
            ImVec2 headerPosMax = popupDrawPos;
            headerPosMax.x += handles->info->metadataWidth;
            headerPosMax.y += 50.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                      ImColor(VkRender::Colors::CRLRed), 9.5f, 0);
            popupDrawPos.y += 40.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                      ImColor(VkRender::Colors::CRLRed), 0.0f, 0);

            ImGui::PushFont(handles->info->font24);
            std::string title = "Custom metadata";
            ImVec2 size = ImGui::CalcTextSize(title.c_str());
            float anchorPoint =
                    (handles->info->metadataWidth - size.x) / 2; // Make a m_Title in center of popup window


            ImGui::Dummy(ImVec2(0.0f, size.y));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 10.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);

            ImGui::SetCursorPosX(anchorPoint);
            ImGui::Text("%s", title.c_str());
            ImGui::PopFont();
            ImGui::Dummy(ImVec2(0.0f, 30.0f));
            ImGui::PopStyleColor();

            static char logName[1024] = "Log no. 1";
            addInputLine("Log name: ", textPadding, textInputPadding, centerWidth, logName);
            ImGui::Dummy(ImVec2(0.0f, 15.0f));

            static char location[1024] = "Test site X";
            addInputLine("Location: ", textPadding, textInputPadding, centerWidth, location);
            ImGui::Dummy(ImVec2(0.0f, 15.0f));

            static char description[1024] = "Offroad navigation";
            addInputLine("Description: ", textPadding, textInputPadding, centerWidth, description);
            ImGui::Dummy(ImVec2(0.0f, 15.0f));

            static char equipment[1024] = "MultiSense mounted on X";
            addInputLine("Equipment Description: ", textPadding, textInputPadding, equipment);
            ImGui::Dummy(ImVec2(0.0f, 15.0f));

            static char camera[1024] = "MultiSense S30"; // TODO Fetch from connection profile
            addInputLine("Camera name: ", textPadding, textInputPadding, centerWidth, camera);
            ImGui::Dummy(ImVec2(0.0f, 15.0f));

            static char collector[1024] = "John Doe";
            addInputLine("Collector name: ", textPadding, textInputPadding, centerWidth, collector);
            ImGui::Dummy(ImVec2(0.0f, 15.0f));

            static char tags[1024] = "self driving, test site x, multisense";
            addInputLine("Tags: ", textPadding, textInputPadding, centerWidth, tags);


            std::string txtLabel = "Custom input field";
            ImGui::Dummy(ImVec2(centerWidth - ImGui::CalcTextSize(txtLabel.c_str()).x / 2.0f, 0.0f));
            ImGui::SameLine();
            ImGui::Text(txtLabel.c_str());
            ImGui::SameLine();
            ImGui::HelpMarker(
                    "Custom input field in JSON format. Use Key=Value such as in the examples below. The '=' symbol is reserved as separator Put each new entry on a newline");
            static char text[1024 * 16] =
                    "road_type = Mountain Road\n"
                    "weather_conditions = Clear sky in the beginning, started to raind around midday followed by harsh winds\n"
                    "project_code = IRAD-2023\n\n\n\n";

            static ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;
            ImGui::Dummy(ImVec2(textPadding / 2.0f, 0.0f));
            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            ImGui::InputTextMultiline("##source", text, IM_ARRAYSIZE(text),
                                      ImVec2(handles->info->metadataWidth - textPadding,
                                             ImGui::GetTextLineHeight() * 10), flags);
            ImGui::PopStyleColor();

            ImGui::PopStyleVar();
            ImGui::SetCursorPos(ImVec2(handles->info->metadataWidth / 2.0f, handles->info->metadataHeight - 50.0f));
            if (ImGui::Button("Set"))
                ImGui::CloseCurrentPopup();

            ImGui::EndPopup();
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(5); // popup style vars
    }

    void addInputLine(std::string description, const float& textPadding, const float& textInputPadding, const float& centerWidth, char* buffer){
        ImGui::Dummy(ImVec2(textPadding, 0.0f));
        ImGui::SameLine();
        std::string label = description;
        ImGui::Text(label.c_str());
        ImGui::SameLine(0.0f, textInputPadding - ImGui::CalcTextSize(label.c_str()).x);
        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
        ImGui::SetNextItemWidth(centerWidth - (textPadding / 2.0f));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 3.0f);
        ImGui::InputText(("##" + description).c_str(), buffer, 1024, 0);
        ImGui::PopStyleColor();


    }
    /** Called once upon this object destruction **/
    void onDetach() override {

    }
};

DISABLE_WARNING_POP

#endif //MULTISENSE_VIEWER_CUSTOMMETADATA_H
