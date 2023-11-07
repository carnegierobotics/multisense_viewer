//
// Created by mgjer on 20/10/2023.
//

#ifndef MULTISENSE_VIEWER_LAYERUTILS_H
#define MULTISENSE_VIEWER_LAYERUTILS_H

#ifdef WIN32
#include <windows.h>
#include <shobjidl.h>
#include <locale>
#include <codecvt>

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


#ifdef WIN32

    static inline std::string selectFolder() {
        PWSTR path = NULL;
        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileOpenDialog* pfd;
            hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileOpenDialog, reinterpret_cast<void**>(&pfd));
            if (SUCCEEDED(hr)) {
                DWORD dwOptions;
                hr = pfd->GetOptions(&dwOptions);
                if (SUCCEEDED(hr)) {
                    hr = pfd->SetOptions(dwOptions | FOS_PICKFOLDERS);
                    if (SUCCEEDED(hr)) {
                        hr = pfd->Show(NULL);
                        if (SUCCEEDED(hr)) {
                            IShellItem* psi;
                            hr = pfd->GetResult(&psi);
                            if (SUCCEEDED(hr)) {
                                hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &path);
                                psi->Release();
                            }
                        }
                    }
                }
                pfd->Release();
            }
            CoUninitialize();
        }

        if (path) {
            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, NULL, NULL);
            std::string str(count - 1, 0);
            WideCharToMultiByte(CP_UTF8, 0, path, -1, &str[0], count, NULL, NULL);
            CoTaskMemFree(path);
            return str;
        }
        return std::string();
    }

    static inline std::string selectYamlFile() {
        PWSTR path = NULL;
        std::string filePath;
        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileOpenDialog* pfd;
            hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileOpenDialog, reinterpret_cast<void**>(&pfd));
            if (SUCCEEDED(hr)) {
                // Set the file types to display only. Notice the double null termination.
                COMDLG_FILTERSPEC rgSpec[] = {
                        { L"YAML Files", L"*.yaml;*.yml" },
                        { L"All Files", L"*.*" },
                };
                pfd->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);

                // Show the dialog
                hr = pfd->Show(NULL);
                if (SUCCEEDED(hr)) {
                    IShellItem* psi;
                    hr = pfd->GetResult(&psi);
                    if (SUCCEEDED(hr)) {
                        hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &path);
                        psi->Release();
                        if (SUCCEEDED(hr)) {
                            // Convert the selected file path to a narrow string
                            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, NULL, NULL);
                            filePath.resize(count - 1);
                            WideCharToMultiByte(CP_UTF8, 0, path, -1, &filePath[0], count, NULL, NULL);
                        }
                    }
                }
                pfd->Release();
            }
            CoUninitialize();
        }

        if (path) {
            CoTaskMemFree(path);
        }

        return filePath;
    }

#else

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
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), Utils::getSystemHomePath().c_str());

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
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), Utils::getSystemHomePath().c_str());

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
#endif
}


#endif //MULTISENSE_VIEWER_LAYERUTILS_H
