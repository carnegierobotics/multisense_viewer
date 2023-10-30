//
// Created by mgjer on 20/10/2023.
//

#ifndef MULTISENSE_VIEWER_LAYERUTILS_H
#define MULTISENSE_VIEWER_LAYERUTILS_H

#ifdef WIN32

#else
#include <gtk/gtk.h>
#endif

#include <imgui.h>


#include "Viewer/ImGui/Widgets.h"

namespace VkRender::LayerUtils {

    struct WidgetPosition {
        float paddingX = -1.0f;
        ImVec4 textColor = VkRender::Colors::CRLTextWhite;
        bool sameLine = false;
    };

    static inline void
    createWidgets(VkRender::GuiObjectHandles *handles, const std::string &area, WidgetPosition pos = WidgetPosition()) {
        for (const auto &elem: Widgets::make()->elements[area]) {

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
                    ImGui::SetNextItemWidth(300.0f);
                    if (ImGui::SliderFloat(elem.label.c_str(), elem.value, elem.minValue, elem.maxValue) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_FLOAT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_INT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(300.0f);
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
                    ImGui::SetNextItemWidth(200.0f);
                    ImGui::InputText(elem.label.c_str(), elem.buf, 1024, 0);
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_BUTTON:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    *elem.button = ImGui::Button(elem.label.c_str());
                    ImGui::PopStyleColor();
                    break;

                default:
                    break;
            }
            if (pos.sameLine)
                ImGui::SameLine();
        }
        ImGui::Dummy(ImVec2());
    }



    static inline std::string selectYamlFile() {
        std::string filename = "";

        gtk_init(0, NULL);

        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                "Open YAML File",
                NULL,
                GTK_FILE_CHOOSER_ACTION_OPEN,
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Open"), GTK_RESPONSE_ACCEPT,
                NULL
        );

        GtkFileFilter *filter = gtk_file_filter_new();
        gtk_file_filter_set_name(filter, "YML Files");
        gtk_file_filter_add_pattern(filter, "*.yml");
        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), Utils::getSysteHomePath().c_str());

        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char *selected_file = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            filename = selected_file;
            g_free(selected_file);
        }

        gtk_widget_destroy(dialog);
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }

        return filename;
    }


    static inline std::string selectFolder() {
        std::string folderPath = "";

        gtk_init(0, NULL);

        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                "Select Folder",
                NULL,
                GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Open"), GTK_RESPONSE_ACCEPT,
                NULL
        );
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), Utils::getSysteHomePath().c_str());

        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char *selected_folder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            folderPath = selected_folder;
            g_free(selected_folder);
        }

        gtk_widget_destroy(dialog);
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }

        return folderPath;
    }

}


#endif //MULTISENSE_VIEWER_LAYERUTILS_H
