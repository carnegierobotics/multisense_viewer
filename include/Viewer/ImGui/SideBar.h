//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_SIDEBAR_H
#define MULTISENSE_SIDEBAR_H

#ifdef WIN32

#include "Viewer/Tools/ReadSharedMemory.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <shellapi.h>

#define AutoConnectReader ReaderWindows
#define elevated() Utils::hasAdminRights()
#else

#include <unistd.h>
#include <sys/types.h>
#include "Viewer/Tools/ReadSharedMemory.h"

#define AutoConnectReader ReaderLinux
#define elevated() getuid()
#endif

#include <algorithm>
#include <queue>
#include <imgui/imgui_internal.h>
#include <sys/types.h>
#include <shellapi.h>

#include "Viewer/Tools/Utils.h"
#include "Viewer/ImGui/Custom/imgui_user.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/AdapterUtils.h"

#define ONE_SECOND 1

// TODO Really long and confusing class. But it handles the sidebar and the pop up modal connect device
class SideBar : public VkRender::Layer {
public:

    typedef enum LOG_COLOR {
        LOG_COLOR_GRAY = 0,
        LOG_COLOR_GREEN = 1,
        LOG_COLOR_RED = 2
    } LOG_COLOR;

    VkRender::EntryConnectDevice m_Entry;
    std::vector<VkRender::EntryConnectDevice> entryConnectDeviceList;

    uint32_t gifFrameIndex = 0;
    uint32_t gifFrameIndex2 = 0;
    std::chrono::steady_clock::time_point gifFrameTimer;
    std::chrono::steady_clock::time_point gifFrameTimer2;
    std::chrono::steady_clock::time_point searchingTextAnimTimer;
    std::chrono::steady_clock::time_point searchNewAdaptersManualConnectTimer;

    std::vector<Utils::Adapter> manualConnectAdapters;

    std::string dots;
    bool btnConnect = false;
    bool btnAdd = false;

    bool enableConnectButton = true;
    enum {
        VIRTUAL_CONNECT = 0,
        MANUAL_CONNECT = 1,
        AUTO_CONNECT = 2
    };

    // Create global object for convenience in other functions
    std::vector<std::string> interfaceNameList;
    std::vector<uint32_t> indexList;

    int connectMethodSelector = 3;
    ImGuiTextBuffer Buf;
    ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
    ImVector<LOG_COLOR> colors;
    ImVec4 lastLogTextColor = VkRender::TextColorGray;
    uint32_t ethernetComboIndex = 0;
    size_t resultsComboIndex = -1;
    bool btnRestartAutoConnect = false;
    bool showRestartButton = false;

    bool startedAutoConnect = false;
    std::unique_ptr<AutoConnectReader> reader;
#ifdef __linudex__
    FILE *autoConnectProcess = nullptr;
#else

    SHELLEXECUTEINFOA shellInfo;
#endif
    std::string btnLabel = "Start";

    void onAttach() override {
        gifFrameTimer = std::chrono::steady_clock::now();
        gifFrameTimer2 = std::chrono::steady_clock::now();
    }

    void onFinishedRender() override {
    }

    void onDetach() override {

#ifdef __linux__
        if (autoConnectProcess != nullptr) {
            if (reader)
                reader->sendStopSignal();
            pclose(autoConnectProcess);
#else
        if (shellInfo.hProcess != nullptr) {
            if (reader)
                reader->sendStopSignal();
            if (TerminateProcess(shellInfo.hProcess, 1) != 0) {
                Log::Logger::getInstance()->info("Terminated AutoConnect program");
            } else
                Log::Logger::getInstance()->info("Failed to terminate AutoConnect program");

#endif
        }
    }

    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth, handles->info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLGray424Main);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 10.0f));
        ImGui::Begin("SideBar", &pOpen, window_flags);


        addPopup(handles);

        ImGui::Spacing();
        ImGui::Spacing();

        if (!handles->devices.empty())
            sidebarElements(handles);


        addDeviceButton(handles);
        ImGui::End();
        ImGui::PopStyleColor(); // bg color
        ImGui::PopStyleVar(2);
    }

private:


    void addLogLine(LOG_COLOR color, const char *fmt, ...) IM_FMTARGS(3) {
        int old_size = Buf.size();
        va_list args;
                va_start(args, fmt);
        Buf.appendfv(fmt, args);
                va_end(args);
        for (int new_size = Buf.size(); old_size < new_size; old_size++)
            if (Buf[old_size] == '\n') {
                LineOffsets.push_back(old_size + 1);
            }

        colors.push_back(color);
    }



    /*
    static void onCameraDetected(AutoConnect::Result res, void *ctx) {
        auto *app = static_cast<SideBar *>(ctx);
        crl::multisense::system::DeviceInfo info;
        crl::multisense::Status status = app->autoConnect.getCameraChannel()->getDeviceInfo(info);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info(
                    "AUTOCONNECT: Found Camera on IP: {}, using Adapter: {}, adapter long m_Name: {}, Camera returned m_Name {}",
                    res.cameraIpv4Address.c_str(), res.networkAdapter.c_str(), res.networkAdapterLongName.c_str(),
                    info.name.c_str());

            // Same IP on same adapter is treated as the same camera
            bool cameraExists = false;
            for (const auto &e: app->entryConnectDeviceList) {
                if (e.IP == res.cameraIpv4Address && e.interfaceIndex == res.index)
                    cameraExists = true;
            }
            if (!cameraExists) {
                VkRender::EntryConnectDevice entry{res.cameraIpv4Address, res.networkAdapter, info.name, res.index,
                                                   res.description};
                app->entryConnectDeviceList.push_back(entry);
                app->resultsComboIndex = app->entryConnectDeviceList.size() - 1;
            }
        } else {
            Log::Logger::getInstance()->info("Failed to fetch camera m_Name from VkRender m_Device");
        }
    }

     */
    /**@brief Function to manage Auto connect with GUI updates
     * Requirements:
     * Should run once popup modal is opened
     * Periodically check the status of connected ethernet devices
     * - The GUI
     * */
    void autoDetectCamera() {
        if (!reader)
            return;

        if (startedAutoConnect) {
            // Try to open reader. TODO timeout or number of tries for opening
            if (!reader->isOpen)
                reader->open();
            else {
                if (reader->read()) {
                    std::string str = reader->getLogLine();
                    if (!str.empty())
                        addLogLine(LOG_COLOR_GRAY, "%s", str.c_str());

                    if (reader->stopRequested) {
                        reader->sendStopSignal();
#ifdef __linux__

                        if (autoConnectProcess != nullptr && pclose(autoConnectProcess) == 0) {
                            startedAutoConnect = false;
                            reader.reset();
                            Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                            autoConnectProcess = nullptr;
                            btnLabel = "Start";
                            return;
                        }
#else
                        if (shellInfo.hProcess != nullptr) {
                            if (TerminateProcess(shellInfo.hProcess, 1) != 0) {
                                Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                            }
                            shellInfo.hProcess = nullptr;
                            reader.reset();
                            btnLabel = "Start";
                            return;
                        }
#endif
                        }

                        // Same IP on same adapter is treated as the same camera
                        VkRender::EntryConnectDevice entry = reader->getResult();
                        if (!entry.cameraName.empty()) {
                            bool cameraExists = false;
                            for (const auto &e: entryConnectDeviceList) {
                                if (e.IP == entry.IP && e.interfaceIndex == entry.interfaceIndex)
                                    cameraExists = true;
                            }
                            if (!cameraExists) {
                                VkRender::EntryConnectDevice e{entry.IP, entry.interfaceName, entry.cameraName,
                                                               entry.interfaceIndex,
                                                               entry.description};
                                entryConnectDeviceList.push_back(e);
                                resultsComboIndex = entryConnectDeviceList.size() - 1;
                            }
                        }
                    }
                }
            } else {
                std::string fileName = Utils::getAssetsPath() + "Generated/StartAutoConnect.sh";
#ifdef __linux__
                autoConnectProcess = popen((fileName).c_str(), "r");
                            if (autoConnectProcess == nullptr) {
                    Log::Logger::getInstance()->info("Failed to start new process, error: %s", strerror(errno));
                } else {
                    startedAutoConnect = true;
                }

#else
                shellInfo.lpVerb = "runas";
                shellInfo.cbSize = sizeof(SHELLEXECUTEINFO);
                shellInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
                shellInfo.hwnd = NULL;
                shellInfo.lpFile = "notepad.exe";
                shellInfo.lpParameters = "-i on -c on";
                shellInfo.lpDirectory = NULL;
                shellInfo.nShow = SW_SHOW;
                shellInfo.hInstApp = NULL;

                bool instance = ShellExecuteExA(&shellInfo);
                if (!instance) {
                    Log::Logger::getInstance()->info("Failed to start new process, error: %zu", GetLastError());
                    reader.reset();
                    btnLabel = "Start";
                    // Reset attempts
                } else {
                    startedAutoConnect = true;
                }

#endif

            }

        }

        void createDefaultElement(VkRender::GuiObjectHandles *handles, const VkRender::EntryConnectDevice &entry) {
            VkRender::Device el{};

            el.name = entry.profileName;
            el.IP = entry.IP;
            el.state = AR_STATE_JUST_ADDED;
            Log::Logger::getInstance()->info("Set dev {}'s state to AR_STATE_JUST_ADDED ", el.name);
            el.interfaceName = entry.interfaceName;
            el.interfaceDescription = entry.description;
            el.clicked = true;
            el.interfaceIndex = entry.interfaceIndex;
            el.isRemoteHead = entry.isRemoteHead;
            handles->devices.push_back(el);

            Log::Logger::getInstance()->info("Connect clicked for Default Device");
            Log::Logger::getInstance()->info("Using: Ip: {}, and profile: {} for {}", entry.IP, entry.profileName,
                                             entry.description);
        }


        void sidebarElements(VkRender::GuiObjectHandles *handles) {
            auto &devices = handles->devices;
            for (size_t i = 0; i < devices.size(); ++i) {
                auto &e = devices.at(i);
                std::string buttonIdentifier;
                ImVec4 btnColor{};
                // Set colors based on state
                switch (e.state) {
                    case AR_STATE_CONNECTED:
                        break;
                    case AR_STATE_CONNECTING:
                        buttonIdentifier = "Connecting";
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLBlueIsh);
                        btnColor = VkRender::CRLBlueIsh;
                        break;
                    case AR_STATE_ACTIVE:
                        buttonIdentifier = "Active";
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray421);
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.42f, 0.31f, 1.0f));
                        btnColor = ImVec4(0.26f, 0.42f, 0.31f, 1.0f);
                        break;
                    case AR_STATE_INACTIVE:
                        buttonIdentifier = "Inactive";
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);
                        btnColor = VkRender::CRLRed;
                        break;
                    case AR_STATE_LOST_CONNECTION:
                        buttonIdentifier = "Lost connection...";
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLBlueIsh);
                        btnColor = VkRender::CRLBlueIsh;
                        break;
                    case AR_STATE_DISCONNECTED:
                        buttonIdentifier = "Disconnected";
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);
                        btnColor = VkRender::CRLRed;
                        break;
                    case AR_STATE_UNAVAILABLE:
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                        ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLDarkGray425);
                        btnColor = VkRender::CRLDarkGray425;
                        buttonIdentifier = "Unavailable";
                        break;
                    case AR_STATE_JUST_ADDED:
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                        btnColor = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
                        buttonIdentifier = "Added...";
                        break;
                    case AR_STATE_DISCONNECT_AND_FORGET:
                    case AR_STATE_REMOVE_FROM_LIST:
                    case AR_STATE_RESET:
                        buttonIdentifier = "Resetting";
                        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                        break;
                }

                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                std::string winId = e.name + "Child";
                ImGui::BeginChild(winId.c_str(), ImVec2(handles->info->sidebarWidth, handles->info->elementHeight),
                                  false, ImGuiWindowFlags_NoDecoration);


                // Stop execution here
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLBlueIsh);

                // Delete a profile
                {
                    // Check if we are currently trying to connect.
                    bool busy = false;
                    for (const auto &dev: handles->devices) {
                        if (dev.state == AR_STATE_CONNECTING)
                            busy = true;
                    }
                    if (ImGui::SmallButton("X") && !busy) {
                        // delete and disconnect devices
                        handles->devices.at(i).state = AR_STATE_DISCONNECT_AND_FORGET;
                        Log::Logger::getInstance()->info("Set dev {}'s state to AR_STATE_DISCONNECT_AND_FORGET ",
                                                         handles->devices.at(i).name);
                        ImGui::PopStyleVar();
                        ImGui::PopStyleColor(3);
                        ImGui::EndChild();
                        continue;
                    }
                }
                ImGui::PopStyleColor();

                ImGui::SetCursorPos(ImVec2(0.0f, ImGui::GetCursorPosY()));

                ImGui::SameLine();
                ImGui::Dummy(ImVec2(0.0f, 20.0f));


                ImVec2 window_pos = ImGui::GetWindowPos();
                ImVec2 window_size = ImGui::GetWindowSize();
                ImVec2 window_center = ImVec2(window_pos.x + window_size.x * 0.5f, window_pos.y + window_size.y * 0.5f);
                ImVec2 cursorPos = ImGui::GetCursorPos();
                ImVec2 lineSize;
                // Profile Name
                {
                    ImGui::PushFont(handles->info->font24);
                    lineSize = ImGui::CalcTextSize(e.name.c_str());
                    cursorPos.x = window_center.x - (lineSize.x / 2);
                    ImGui::SetCursorPos(cursorPos);
                    ImGui::Text("%s", e.name.c_str());
                    ImGui::PopFont();
                }
                // Camera Name
                {
                    ImGui::PushFont(handles->info->font13);
                    lineSize = ImGui::CalcTextSize(e.cameraName.c_str());
                    cursorPos.x = window_center.x - (lineSize.x / 2);
                    ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
                    ImGui::Text("%s", e.cameraName.c_str());
                    ImGui::PopFont();
                }
                // Camera IP Address
                {
                    ImGui::PushFont(handles->info->font13);
                    lineSize = ImGui::CalcTextSize(e.IP.c_str());
                    cursorPos.x = window_center.x - (lineSize.x / 2);
                    ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));

                    ImGui::Text("%s", e.IP.c_str());
                    ImGui::PopFont();
                }

                // Status Button
                {
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::PushFont(handles->info->font18);
                    //ImGuiStyle style = ImGui::GetStyle();
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
                    cursorPos.x = window_center.x - (ImGui::GetFontSize() * 10 / 2);
                    ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
                }

                //buttonIdentifier += "##" + e.IP;
                //e.clicked = ImGui::Button(buttonIdentifier.c_str(), ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2));

                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 0.3f);       // No tint

                // Check if other device is already attempting to connect
                bool busy = false;
                for (const auto &dev: handles->devices) {
                    if (dev.state == AR_STATE_CONNECTING)
                        busy = true;
                }

                // Connect button
                if (buttonIdentifier == "Connecting") {
                    e.clicked = ImGui::ButtonWithGif(buttonIdentifier.c_str(), ImVec2(ImGui::GetFontSize() * 10, 35.0f),
                                                     handles->info->gif.image[gifFrameIndex2], ImVec2(35.0f, 35.0f),
                                                     uv0,
                                                     uv1,
                                                     tint_col, btnColor) && !busy;
                } else {
                    e.clicked = ImGui::Button(buttonIdentifier.c_str(),
                                              ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2)) && !busy;
                }
                auto time = std::chrono::steady_clock::now();
                auto time_span =
                        std::chrono::duration_cast<std::chrono::duration<float>>(time - gifFrameTimer2);

                if (time_span.count() > ((float) *handles->info->gif.delay) / 1000.0f) {
                    gifFrameTimer2 = std::chrono::steady_clock::now();
                    gifFrameIndex2++;
                }

                if (gifFrameIndex2 >= handles->info->gif.totalFrames)
                    gifFrameIndex2 = 0;

                ImGui::PopFont();
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);

                ImGui::EndChild();
            }
        }

        void addDeviceButton(VkRender::GuiObjectHandles *handles) {

            ImGui::SetCursorPos(ImVec2((handles->info->sidebarWidth / 2) - (handles->info->addDeviceWidth / 2),
                                       handles->info->height - handles->info->addDeviceBottomPadding));

            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLBlueIsh);
            btnAdd = ImGui::Button("ADD DEVICE", ImVec2(handles->info->addDeviceWidth, handles->info->addDeviceHeight));

            ImGui::PopStyleColor();
            if (btnAdd) {
                ImGui::OpenPopup("add_device_modal");
            }

            ImGui::SetCursorPos(ImVec2((handles->info->sidebarWidth / 2) - (handles->info->addDeviceWidth / 2),
                                       handles->info->height - handles->info->addDeviceBottomPadding +
                                       handles->info->addDeviceHeight + 25.0f));

            if (ImGui::Button("Show Debug")) {
                handles->showDebugWindow = !handles->showDebugWindow;
            }
        }

        void addPopup(VkRender::GuiObjectHandles *handles) {
            ImGui::SetNextWindowSize(ImVec2(handles->info->popupWidth, handles->info->popupHeight), ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
            ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::CRLCoolGray);


            if (ImGui::BeginPopupModal("add_device_modal", nullptr,
                                       ImGuiWindowFlags_NoDecoration)) {
                /** HEADER FIELD */
                ImVec2 popupDrawPos = ImGui::GetCursorScreenPos();

                ImVec2 headerPosMax = popupDrawPos;
                headerPosMax.x += handles->info->popupWidth;
                headerPosMax.y += 50.0f;
                ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                          ImColor(VkRender::CRLRed), 10.0f, 0);

                ImGui::PushFont(handles->info->font24);
                std::string title = "Connect to MultiSense";
                ImVec2 size = ImGui::CalcTextSize(title.c_str());
                float anchorPoint =
                        (handles->info->popupWidth - size.x) / 2; // Make a m_Title in center of popup window


                ImGui::Dummy(ImVec2(0.0f, size.y));

                ImGui::SetCursorPosX(anchorPoint);
                ImGui::Text("%s", title.c_str());
                ImGui::PopFont();
                //ImGui::Separator();
                //ImGui::InputText("Profile m_Name##1", inputName.data(),inputFieldNameLength);

                ImGui::Dummy(ImVec2(0.0f, 25.0f));

                /** PROFILE NAME FIELD */

                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Text("1. Profile Name:");
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::CRLDarkGray425);
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                ImGui::CustomInputTextWithHint("##InputProfileName", "MultiSense Profile", &m_Entry.profileName,
                                               ImGuiInputTextFlags_AutoSelectAll);
                ImGui::Dummy(ImVec2(0.0f, 30.0f));


                /** SELECT METHOD FOR CONNECTION FIELD */
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Text("2. Select method for connection:");
                ImGui::PopStyleColor();
                ImGui::Dummy(ImVec2(0.0f, 5.0f));

                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 0.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(10.0f, 0.0f));

                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
                //ImGui::BeginChild("IconChild", ImVec2(handles->info->popupWidth, 40.0f), false, ImGuiWindowFlags_NoDecoration);

                (ImGui::ImageButtonText("Automatic", &connectMethodSelector, AUTO_CONNECT, ImVec2(190.0f, 55.0f),
                                        handles->info->imageButtonTextureDescriptor[3], ImVec2(33.0f, 31.0f), uv0, uv1,
                                        tint_col));


                ImGui::SameLine(0, 25.0f);
                (ImGui::ImageButtonText("Manual", &connectMethodSelector, MANUAL_CONNECT, ImVec2(190.0f, 55.0f),
                                        handles->info->imageButtonTextureDescriptor[4], ImVec2(40.0f, 40.0f), uv0, uv1,
                                        tint_col));


                /*
                             ImGui::SameLine(0, 30.0f);
                (ImGui::ImageButtonText("Virtual", &connectMethodSelector, VIRTUAL_CONNECT, ImVec2(125.0f, 55.0f),
                                        handles->info->imageButtonTextureDescriptor[5], ImVec2(40.0f, 40.0f), uv0, uv1,
                                        bg_col,
                                        tint_col));
                 */
                ImGui::PopStyleVar();
                //ImGui::EndChild();

                ImGui::PopStyleColor(); // ImGuiCol_FrameBg
                ImGui::PopStyleVar(2); // RadioButton

                autoDetectCamera();

                /** AUTOCONNECT FIELD BEGINS HERE*/
                if (connectMethodSelector == AUTO_CONNECT) {
                    m_Entry.cameraName = "AutoConnect";
                    ImGui::Dummy(ImVec2(0.0f, 12.0f));
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();

                    if (ImGui::Button(btnLabel.c_str(), ImVec2(80.0f, 20.0f))) {
                        if (btnLabel == "Start") {
                            reader = std::make_unique<AutoConnectReader>();
                            btnLabel = "Stop";
                        } else {
                            reader->sendStopSignal();
#ifdef __linux__
                            if (autoConnectProcess != nullptr && pclose(autoConnectProcess) == 0) {
                                startedAutoConnect = false;
                                reader.reset();
                                Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                                autoConnectProcess = nullptr;
                                btnLabel = "Start";
                            }
#else
                            if (shellInfo.hProcess != nullptr) {
                                if (TerminateProcess(shellInfo.hProcess, 1) != 0) {
                                    Log::Logger::getInstance()->info("Stopped the auto connect process gracefully");
                                }
                                shellInfo.hProcess = nullptr;
                                reader.reset();
                                btnLabel = "Start";
                                startedAutoConnect = false;

                            }
#endif
                        }
                    }


                    /*
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                    ImGui::Text("Status:");
                    if (showRestartButton) {
                        ImGui::SameLine(0, 15.0f);
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
                        btnRestartAutoConnect = ImGui::SmallButton("Restart?");
                        if (btnRestartAutoConnect) {
                            entryConnectDeviceList.clear();
                            //autoConnect.clearSearchedAdapters();
                            m_Entry.reset();
                            //autoConnect.stopAutoConnect();
                            showRestartButton = false;
                        }
                        ImGui::PopStyleColor();
                    }
                    ImGui::PopStyleColor();
                    */

                    /** STATUS SPINNER */
                    ImGui::SameLine();
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 12.0f);
                    // Create child window regardless of gif spinner state in order to keep cursor m_Position constant
                    ImGui::BeginChild("Gif viewer", ImVec2(40.0f, 40.0f), false, ImGuiWindowFlags_NoDecoration);
                    if (startedAutoConnect)
                        addSpinnerGif(handles);
                    ImGui::EndChild();

                    ImGui::SameLine(0, 250.0f);
                    ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::CRLBlueIsh);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                    ImGui::HelpMarker(" If no packet at adapter is received try the following: \n "
                                      " 1. Reconnect ethernet cables \n "
                                      " 2. Power cycle the camera \n "
                                      " 3. Wait 20-30 seconds. If no packet is received contact support  \n\n");

                    ImGui::PopStyleColor(2);

                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLDarkGray425);
                    const char *id = "Log Window";
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));

                    ImGui::SameLine();
                    ImGui::BeginChild(id, ImVec2(handles->info->popupWidth - 40.0f, 85.0f), false, 0);
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0));
                    const char *buf = Buf.begin();
                    const char *buf_end = Buf.end();
                    ImGuiListClipper clipper;
                    clipper.Begin(LineOffsets.Size);
                    while (clipper.Step()) {
                        for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++) {
                            const char *line_start = buf + LineOffsets[line_no];
                            const char *line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] -
                                                                                       1)
                                                                                    : buf_end;
                            // Weird index magic. colors is an ImGui vector.
                            if (line_no > 0 && line_no < colors.size() - 1) {
                                LOG_COLOR col = colors[line_no + 1];
                                switch (col) {
                                    case LOG_COLOR_GRAY:
                                        lastLogTextColor = VkRender::TextColorGray;
                                        break;
                                    case LOG_COLOR_GREEN:
                                        lastLogTextColor = VkRender::TextGreenColor;
                                        break;
                                    case LOG_COLOR_RED:
                                        lastLogTextColor = VkRender::TextRedColor;
                                        break;
                                    default:
                                        lastLogTextColor = VkRender::TextColorGray;
                                        break;
                                }
                            }
                            ImGui::PushStyleColor(ImGuiCol_Text, lastLogTextColor);
                            ImGui::Dummy(ImVec2(5.0f, 0.0f));
                            ImGui::SameLine();
                            ImGui::TextUnformatted(line_start, line_end);
                            ImGui::PopStyleColor();

                        }
                    }
                    clipper.End();
                    ImGui::PopStyleColor();


                    ImGui::PopStyleVar();
                    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                        ImGui::SetScrollHereY(1.0f);
                    ImGui::EndChild();
                    ImGui::Dummy(ImVec2(0.0f, 30.0f));
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();

                    auto time = std::chrono::steady_clock::now();
                    std::chrono::duration<float> time_span =
                            std::chrono::duration_cast<std::chrono::duration<float>>(time - searchingTextAnimTimer);

                    if (time_span.count() > 0.35f && startedAutoConnect) {
                        searchingTextAnimTimer = std::chrono::steady_clock::now();
                        dots.append(".");

                        if (dots.size() > 4)
                            dots.clear();

                    } else if (!startedAutoConnect)
                        dots = ".";

                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);

                    if (entryConnectDeviceList.empty() && startedAutoConnect) {
                        ImGui::Text("%s", ("Searching" + dots).c_str());
                    } else if (entryConnectDeviceList.empty())
                        ImGui::Text("Searching...");
                    else {
                        ImGui::Text("Select:");
                    }

                    ImGui::PopStyleColor();
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    static bool selected = false;

                    if (!selected) {
                        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4());
                    } else {
                        ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyle().Colors[ImGuiCol_Header]);
                    }
                    // Header
                    // HeaderHovered
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLDarkGray425);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 5.0f));

                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::BeginChild("##ResultsChild", ImVec2(handles->info->popupWidth - (20.0f * 2.0f), 50.0f), true,
                                      0);
                    for (size_t n = 0; n < entryConnectDeviceList.size(); n++) {
                        if (entryConnectDeviceList[n].cameraName.empty()) continue;
                        if (ImGui::Selectable((entryConnectDeviceList[n].cameraName + "##" + std::to_string(n)).c_str(),
                                              resultsComboIndex == n,
                                              ImGuiSelectableFlags_DontClosePopups,
                                              ImVec2(handles->info->popupWidth - (20.0f * 2), 15.0f))) {

                            resultsComboIndex = n;
                            entryConnectDeviceList[n].profileName = entryConnectDeviceList[n].cameraName; // Keep profile m_Name if user inputted this before auto-connect is finished
                            m_Entry = entryConnectDeviceList[n];
                            selected = true;

                        }
                    }
                    ImGui::EndChild();
                    ImGui::PopStyleColor(2);
                    ImGui::PopStyleVar();

                    ImGui::Dummy(ImVec2(20.0f, 10.0));
                    ImGui::Dummy(ImVec2(20.0f, 0.0));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                    ImGui::Checkbox(" Remote Head", &m_Entry.isRemoteHead);
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::CRLBlueIsh);
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextWhite);
                    ImGui::HelpMarker("\n  Check this if you are connecting to a remote head device  \n ");
                    ImGui::PopStyleColor(3);

                }
                    /** MANUAL_CONNECT FIELD BEGINS HERE*/
                else if (connectMethodSelector == MANUAL_CONNECT) {
                    {
                        ImGui::Dummy(ImVec2(20.0f, 0.0f));
                        ImGui::SameLine();
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                        ImGui::Text("Camera IP:");
                        ImGui::PopStyleColor();
                        ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    }

                    ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::CRLDarkGray425);
                    ImGui::Dummy(ImVec2(20.0f, 5.0f));
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                    ImGui::CustomInputTextWithHint("##inputIP", "Default: 10.66.171.21", &m_Entry.IP,
                                                   ImGuiInputTextFlags_CharsScientific |
                                                   ImGuiInputTextFlags_AutoSelectAll |
                                                   ImGuiInputTextFlags_CharsNoBlank);
                    ImGui::Dummy(ImVec2(0.0f, 10.0f));

                    {
                        ImGui::Dummy(ImVec2(20.0f, 0.0f));
                        ImGui::SameLine(0.0f, 10.0f);
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                        ImGui::Text("Select network adapter:");
                        ImGui::PopStyleColor();
                        ImGui::SameLine(0.0f, 10.0f);
                    }

                    ImGui::Dummy(ImVec2(0.0f, 5.0f));


                    // Call once a second
                    auto time = std::chrono::steady_clock::now();
                    std::chrono::duration<float> time_span =
                            std::chrono::duration_cast<std::chrono::duration<float>>(
                                    time - searchNewAdaptersManualConnectTimer);
                    if (time_span.count() > ONE_SECOND) {
                        searchNewAdaptersManualConnectTimer = std::chrono::steady_clock::now();

                        manualConnectAdapters = Utils::listAdapters();

                        interfaceNameList.clear();
                        indexList.clear();
                    }

                    // immediate mode vector item ordering
                    // -- requirements --
                    // Always have a base item in it.
                    // if we push back other items then remove base item
                    // No identical items
                    for (const auto &a: manualConnectAdapters) {
                        if (a.supports && !Utils::isInVector(interfaceNameList, a.ifName)) {
                            interfaceNameList.push_back(a.ifName);
                            indexList.push_back(a.ifIndex);

                            if (Utils::isInVector(interfaceNameList, "No adapters found")) {
                                Utils::delFromVector(&interfaceNameList, std::string("No adapters found"));
                            }
                        }
                    }

                    if (!Utils::isInVector(interfaceNameList, "No adapters found") &&
                        interfaceNameList.empty()) {
                        interfaceNameList.emplace_back("No adapters found");
                        indexList.push_back(0);
                    }

                    m_Entry.interfaceName = interfaceNameList[ethernetComboIndex];  // Pass in the preview value visible before opening the combo (it could be anything)
                    m_Entry.interfaceIndex = indexList[ethernetComboIndex];
                    //if (!adapters.empty())
                    //    m_Entry.interfaceName = adapters[ethernetComboIndex].networkAdapter;
                    static ImGuiComboFlags flags = 0;
                    ImGui::Dummy(ImVec2(20.0f, 5.0f));
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                    if (ImGui::BeginCombo("##SelectAdapter", m_Entry.interfaceName.c_str(), flags)) {
                        for (size_t n = 0; n < interfaceNameList.size(); n++) {
                            const bool is_selected = (ethernetComboIndex == n);
                            if (ImGui::Selectable(interfaceNameList[n].c_str(), is_selected))
                                ethernetComboIndex = static_cast<uint32_t>(n);

                            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::PopStyleColor(); // ImGuiCol_FrameBg
                    m_Entry.cameraName = "Manual";

                    ImGui::Dummy(ImVec2(40.0f, 10.0));
                    ImGui::Dummy(ImVec2(20.0f, 0.0));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                    ImGui::Checkbox("  Configure System Network", &handles->configureNetwork);
                    ImGui::SameLine(0, 20.0f);
                    ImGui::Checkbox(" Remote Head", &m_Entry.isRemoteHead);

                    if (handles->configureNetwork) {
                        if (elevated()) {
                            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLRed);
                            ImGui::Dummy(ImVec2(40.0f, 10.0));
                            ImGui::Dummy(ImVec2(20.0f, 0.0));
                            ImGui::SameLine();
#ifdef WIN32
                            ImGui::Text("Launch app as admin to \nauto configure network adapter");
#else
                            ImGui::Text("Launch app as root to \nauto configure network adapter");
#endif
                            enableConnectButton = false;
                            ImGui::PopStyleColor();
                        }

                    } else
                        enableConnectButton = true;
                    ImGui::PopStyleColor();
                } else {

                }


                /** CANCEL/CONNECT FIELD BEGINS HERE*/
                ImGui::Dummy(ImVec2(0.0f, 40.0f));
                ImGui::SetCursorPos(ImVec2(0.0f, handles->info->popupHeight - 50.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                bool btnCancel = ImGui::Button("cancel", ImVec2(150.0f, 30.0f));
                ImGui::SameLine(0, 110.0f);
                if (!m_Entry.ready(handles->devices, m_Entry) || !enableConnectButton) {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::TextColorGray);
                }
                btnConnect = ImGui::Button("connect", ImVec2(150.0f, 30.0f));
                // If hovered, and no admin rights while auto config is checked, and a connect method must be selected
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) &&
                    (!m_Entry.ready(handles->devices, m_Entry) || !enableConnectButton) &&
                    (connectMethodSelector == AUTO_CONNECT || connectMethodSelector == MANUAL_CONNECT)) {
                    ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::CRLBlueIsh);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
                    ImGui::BeginTooltip();
                    std::vector<std::string> errors = m_Entry.getNotReadyReasons(handles->devices, m_Entry);
                    ImGui::Text("Please solve the following: ");
                    if (elevated()) {
#ifdef WIN32
                        errors.emplace_back("Admin rights is needed to auto configure network");
#else
                        errors.emplace_back("Root access is needed to auto configure network");
#endif
                    }
                    for (size_t i = 0; i < errors.size(); ++i) {
                        ImGui::Text("%s", (std::to_string(i + 1) + ". " + errors[i]).c_str());
                    }
                    ImGui::EndTooltip();
                    ImGui::PopStyleColor();
                    ImGui::PopStyleVar();
                }
                if (!m_Entry.ready(handles->devices, m_Entry) || !enableConnectButton) {
                    ImGui::PopStyleColor();
                    ImGui::PopItemFlag();
                }
                ImGui::Dummy(ImVec2(0.0f, 5.0f));

                if (btnCancel) {
                    ImGui::CloseCurrentPopup();
                }

                if (btnConnect && m_Entry.ready(handles->devices, m_Entry) && enableConnectButton) {
                    createDefaultElement(handles, m_Entry);
                    ImGui::CloseCurrentPopup();
                }

                ImGui::EndPopup();
            }
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(5); // popup style vars
        }

        void addSpinnerGif(VkRender::GuiObjectHandles *handles) {

            ImVec2 size = ImVec2(40.0f,
                                 40.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
            ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
            ImVec2 uv1 = ImVec2(1.0f, 1.0f);

            ImVec4 bg_col = ImVec4(0.054f, 0.137f, 0.231f, 1.0f);
            ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 0.0f);

            ImGui::Image(handles->info->gif.image[gifFrameIndex], size, uv0, uv1, bg_col, tint_col);
            auto time = std::chrono::steady_clock::now();
            std::chrono::duration<float> time_span =
                    std::chrono::duration_cast<std::chrono::duration<float>>(time - gifFrameTimer);

            if (time_span.count() > ((float) *handles->info->gif.delay) / 1000.0f) {
                gifFrameTimer = std::chrono::steady_clock::now();
                gifFrameIndex++;
            }

            if (gifFrameIndex == handles->info->gif.totalFrames)
                gifFrameIndex = 0;
        }


    };


#endif //MULTISENSE_SIDEBAR_H
