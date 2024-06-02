//
// Created by mgjer on 20/10/2023.
//

#ifndef MULTISENSE_VIEWER_LAYERUTILS_H
#define MULTISENSE_VIEWER_LAYERUTILS_H

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shobjidl.h>
#include <locale>
#include <codecvt>

#else
DISABLE_WARNING_PUSH
DISABLE_WARNING_OLD_STYLE_CAST

#include <gtk/gtk.h>

DISABLE_WARNING_POP
#endif

#include <imgui.h>

#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Tools/Macros.h"

namespace VkRender::LayerUtils {

    struct LoadFileInfo {
        std::filesystem::path path;
        std::string fileType;
    };

#ifdef WIN32

    static inline std::filesystem::path selectFolder(std::string openLocation = "") {
        PWSTR path = nullptr;
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileOpenDialog* pfd;
            hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_IFileOpenDialog,
                                  reinterpret_cast<void**>(&pfd));
            if (SUCCEEDED(hr)) {
                DWORD dwOptions;
                hr = pfd->GetOptions(&dwOptions);
                if (SUCCEEDED(hr)) {
                    hr = pfd->SetOptions(dwOptions | FOS_PICKFOLDERS);
                    if (SUCCEEDED(hr)) {
                        hr = pfd->Show(nullptr);
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
            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
            std::string str(count - 1, 0);
            WideCharToMultiByte(CP_UTF8, 0, path, -1, &str[0], count, nullptr, nullptr);
            CoTaskMemFree(path);
            return str;
        }
        return std::string();
    }

    static inline LoadFileInfo selectFile(const std::string& dialogName, const std::string& fileType, const std::string& setCurrentFolder) {
        std::string fileWithDot = "*." + fileType;
        std::wstring fileTypeFilter = std::wstring(fileWithDot.begin(), fileWithDot.end());
        PWSTR path = nullptr;
        std::string filePath;
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileOpenDialog *pfd;
            hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_IFileOpenDialog,
                                  reinterpret_cast<void **>(&pfd));
            if (SUCCEEDED(hr)) {
                // Set the file types to display only. Notice the double null termination.

                COMDLG_FILTERSPEC rgSpec[] = {
                        {L".obj", fileTypeFilter.c_str()},
                        {L"All Files", L"*.*"},
                };
                pfd->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);

                // Set the default folder
                if (!setCurrentFolder.empty()) {
                    IShellItem *psiFolder;
                    std::wstring folderWStr(setCurrentFolder.begin(), setCurrentFolder.end());
                    hr = SHCreateItemFromParsingName(folderWStr.c_str(), nullptr, IID_PPV_ARGS(&psiFolder));
                    if (SUCCEEDED(hr)) {
                        pfd->SetFolder(psiFolder);
                        psiFolder->Release();
                    }
                }

                // Show the dialog
                hr = pfd->Show(nullptr);
                if (SUCCEEDED(hr)) {
                    IShellItem *psi;
                    hr = pfd->GetResult(&psi);
                    if (SUCCEEDED(hr)) {
                        hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &path);
                        psi->Release();
                        if (SUCCEEDED(hr)) {
                            // Convert the selected file path to a narrow string
                            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                            filePath.resize(count - 1);
                            WideCharToMultiByte(CP_UTF8, 0, path, -1, &filePath[0], count, nullptr, nullptr);
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

        return {filePath, fileType};
    }

#else
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_ALL

    static inline LoadFileInfo selectFile(const std::string& dialogName, const std::string& fileType, const std::string& setCurrentFolder) {
        std::string filename;

        gtk_init(0, nullptr);

        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                dialogName.c_str(),
                nullptr,
                GTK_FILE_CHOOSER_ACTION_OPEN,
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Open"), GTK_RESPONSE_ACCEPT,
                nullptr
        );

        GtkFileFilter *filter = gtk_file_filter_new();
        gtk_file_filter_set_name(filter, (fileType + " Files").c_str()); // Set filter name dynamically based on fileType
        gtk_file_filter_add_pattern(filter, ("*." + fileType).c_str()); // Set filter pattern dynamically
        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), setCurrentFolder.c_str());

        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char *selected_file = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            filename = selected_file;
            g_free(selected_file);
        }

        gtk_widget_destroy(dialog);
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }

        return {filename, fileType};
    }


    static inline std::filesystem::path selectFolder(std::string openLocation) {
        std::string folderPath;

        gtk_init(0, nullptr);

        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                "Select Folder",
                nullptr,
                GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Open"), GTK_RESPONSE_ACCEPT,
                nullptr
        );
        std::string openLoc = openLocation.empty() ? Utils::getSystemHomePath().string() : openLocation;
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), openLoc.c_str());

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

    DISABLE_WARNING_POP
#endif

    struct WidgetPosition {
        float paddingX = -1.0f;
        ImVec4 textColor = VkRender::Colors::CRLTextWhite;
        bool sameLine = false;

        float maxElementWidth = 300.0f;
    };

    static inline void
    createWidgets(VkRender::GuiObjectHandles *handles, const ScriptWidgetPlacement &area,
                  WidgetPosition pos = WidgetPosition()) {
        for (auto &elem: Widgets::make()->elements[area]) {
            if (pos.paddingX != -1) {
                ImGui::Dummy(ImVec2(pos.paddingX, 0.0f));
                ImGui::SameLine();
            }

            switch (elem.type) {
                case WIDGET_CHECKBOX:
                    ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
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
                    ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                    ImGui::SetNextItemWidth(pos.maxElementWidth);
                    ImGui::InputText(elem.label.c_str(), elem.buf, 1024, 0);
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_BUTTON:
                    ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                    *elem.button = ImGui::Button(elem.label.c_str());
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_GLM_VEC_3:
                    ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                    ImGui::SetNextItemWidth(35.0f);
                    ImGui::InputText(("x: " + elem.label).c_str(), elem.glm.xBuf, 16, ImGuiInputTextFlags_CharsDecimal);
                    ImGui::SetNextItemWidth(35.0f);
                    ImGui::InputText(("y: " + elem.label).c_str(), elem.glm.yBuf, 16, ImGuiInputTextFlags_CharsDecimal);
                    ImGui::SetNextItemWidth(35.0f);
                    ImGui::InputText(("z: " + elem.label).c_str(), elem.glm.zBuf, 16, ImGuiInputTextFlags_CharsDecimal);
                    try {
                        elem.glm.vec3->x = std::stof(elem.glm.xBuf);
                        elem.glm.vec3->y = std::stof(elem.glm.yBuf);
                        elem.glm.vec3->z = std::stof(elem.glm.zBuf);
                    }
                    catch (...) {
                    }

                    ImGui::PopStyleColor();
                    break;
                case WIDGET_SELECT_DIR_DIALOG:
                    ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);

                    if (ImGui::Button(elem.label.c_str())) {
                        if (!elem.future->valid())
                            *elem.future = std::async(selectFolder, "");
                    }

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

}


#endif //MULTISENSE_VIEWER_LAYERUTILS_H
