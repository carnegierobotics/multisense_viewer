//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_SIDEBAR_H
#define MULTISENSE_SIDEBAR_H

#include "AutoConnect.h"


#ifdef WIN32
#include "AutoConnectWindows.h"
#define AutoConnectHandle AutoConnectWindows
#define elevated() Utils::hasAdminRights()
#else

#include <unistd.h>
#include <sys/types.h>
#include "AutoConnectLinux.h"

#define AutoConnectHandle AutoConnectLinux
#define elevated() getuid()
#endif

#include <MultiSense/Src/Tools/Utils.h>

#include <algorithm>
#include <queue>
#include "imgui_internal.h"
#include "Layer.h"
#include "imgui_user.h"
#include <sys/types.h>

#define ONE_SECOND 1


// TODO Really long and confusing class. But it handles the sidebar and the pop up modal connect device
class SideBar : public VkRender::Layer {
public:

    // Create global object for convenience in other functions
    AutoConnectHandle autoConnect{};
    bool refreshAdapterList = true; // Set to true to find adapters on next call
    std::vector<AutoConnect::Result> adapters;
    std::vector<std::string> interfaceDescriptionList;
    std::vector<uint32_t> indexList;

    VkRender::EntryConnectDevice entry;
    std::vector<VkRender::EntryConnectDevice> entryConnectDeviceList;

    uint32_t gifFrameIndex = 0;
    std::chrono::steady_clock::time_point gifFrameTimer;
    std::chrono::steady_clock::time_point searchingTextAnimTimer;
    std::chrono::steady_clock::time_point searchNewAdaptersManualConnectTimer;

    std::string dots;
    bool btnConnect = false;
    bool btnAdd = false;
    bool skipUserFriendlySleepDelay = false;
    bool dontRunAutoConnect = false;
    bool enableConnectButton = true;
    enum {
        VIRTUAL_CONNECT = 0,
        MANUAL_CONNECT = 1,
        AUTO_CONNECT = 2
    };
    int connectMethodSelector = 3;
    ImGuiTextBuffer Buf;
    ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
    ImVector<int> colors;
    ImVec4 lastLogTextColor = VkRender::TextColorGray;

    uint32_t ethernetComboIndex = 0;
    size_t resultsComboIndex = -1;

    void onAttach() override {
        autoConnect.setDetectedCallback(SideBar::onCameraDetected, this);
        autoConnect.setEventCallback(SideBar::onEvent);
        gifFrameTimer = std::chrono::steady_clock::now();
    }

    void onFinishedRender() override {

    }

    void onDetach() override {

    }


    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        ;


        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth, handles->info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::CRLGray424Main);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 10.0f));
        ImGui::Begin("SideBar", &pOpen, window_flags);

        ImVec2 txtSize = ImGui::CalcTextSize(handles->info->title.c_str());
        float xOffset = (handles->info->sidebarWidth / 2) - (txtSize.x / 2);
        ImGui::Dummy(ImVec2(xOffset, 0.0f));
        ImGui::SameLine();
        ImGui::Text("%s", handles->info->title.c_str());

        txtSize = ImGui::CalcTextSize(handles->info->deviceName.c_str());
        // If it is too long then just remove some word towards the end.
        if (txtSize.x > handles->info->sidebarWidth) {
            std::string devName = handles->info->deviceName;
            while (txtSize.x > handles->info->sidebarWidth) {
                devName.erase(devName.find_last_of(' '), devName.length());
                txtSize = ImGui::CalcTextSize(devName.c_str());
            }
            xOffset = (handles->info->sidebarWidth / 2) - (txtSize.x / 2);
            ImGui::Dummy(ImVec2(xOffset, 0.0f));
            ImGui::SameLine();
            ImGui::Text("%s", devName.c_str());
        } else {
            xOffset = (handles->info->sidebarWidth / 2) - (txtSize.x / 2);
            ImGui::Dummy(ImVec2(xOffset, 0.0f));
            ImGui::SameLine();
            ImGui::Text("%s", handles->info->deviceName.c_str());
        }


        // Update frame time display
        if (handles->info->firstFrame) {
            std::rotate(handles->info->frameTimes.begin(), handles->info->frameTimes.begin() + 1,
                        handles->info->frameTimes.end());
            float frameTime = 1000.0f / (handles->info->frameTimer * 1000.0f);
            handles->info->frameTimes.back() = frameTime;
            if (frameTime < handles->info->frameTimeMin) {
                handles->info->frameTimeMin = frameTime;
            }
            if (frameTime > handles->info->frameTimeMax) {
                handles->info->frameTimeMax = frameTime;
            }
        }

        ImGui::Dummy(ImVec2(5.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PlotLines("##FrameTimes", &handles->info->frameTimes[0], 50, 0, "fps", handles->info->frameTimeMin,
                         handles->info->frameTimeMax, ImVec2(handles->info->sidebarWidth - 28.0f, 80.0f));


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

    void AddLog(int color, const char *fmt, ...) IM_FMTARGS(3) {
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


private:


    static void onEvent(std::string event, void *ctx, int color = 0) {
        auto *app = static_cast<SideBar *>(ctx);
        // added a delay for user-friendliness. Also works great cause switching colors should be done on main thread
        // but Push/Pop style works here because of the delay.
        if (!app->skipUserFriendlySleepDelay) {
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
        if (event == "Finished") {
            app->autoConnect.setShouldProgramClose(true);
            app->dontRunAutoConnect = true;
        }
        app->AddLog(color, "[INFO] %s\n", event.c_str());
    }

    static void onCameraDetected(AutoConnect::Result res, void *ctx) {
        auto *app = static_cast<SideBar *>(ctx);


        crl::multisense::system::DeviceInfo info;
        crl::multisense::Status status = app->autoConnect.getCameraChannel()->getDeviceInfo(info);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info(
                    "AUTOCONNECT: Found Camera on IP: {}, using Adapter: {}, adapter long name: {}, Camera returned name {}",
                    res.cameraIpv4Address.c_str(), res.networkAdapter.c_str(), res.networkAdapterLongName.c_str(),
                    info.name.c_str());

            bool ipExists = false;
            for (const auto &e: app->entryConnectDeviceList) {
                if (e.IP == res.cameraIpv4Address)
                    ipExists = true;
            }

            if (!ipExists) {
                VkRender::EntryConnectDevice entry{res.cameraIpv4Address, res.networkAdapter, info.name, res.index,
                                                   res.description};
                app->entryConnectDeviceList.push_back(entry);
                app->resultsComboIndex = app->entryConnectDeviceList.size() - 1;
            }
        } else {
            Log::Logger::getInstance()->info("Failed to fetch camera name from VkRender device");

        }


    }

    /**@brief Function to manage Auto connect with GUI updates
     * Requirements:
     * Should run once popup modal is opened
     * Periodically check the status of connected ethernet devices
     * - The GUI
     * */
    void autoDetectCamera() {
        // If this method is not selected, and it is running then, run it otherwise stop it.
        // If already running just check if we should close it before opening it again.
        if ((autoConnect.shouldProgramClose() && autoConnect.running) ||
            (connectMethodSelector != AUTO_CONNECT && autoConnect.running)) {
            autoConnect.stop();
            Log::Logger::getInstance()->info("Stopping Auto connect");
        }

        // Reset this variable by re-selecting auto connect button
        // Also clear the searched adapter which will start a new search once the auto connect tab is selected again
        if (connectMethodSelector != AUTO_CONNECT) {
            dontRunAutoConnect = false;
            autoConnect.clearSearchedAdapters();
        }


        // Dont run this feature if we have not selected it in gui
        if (connectMethodSelector != AUTO_CONNECT || autoConnect.running || dontRunAutoConnect)
            return;

        Buf.clear();
        LineOffsets.clear();
        LineOffsets.push_back(0);
        colors.clear();
        colors.push_back(0);

        AddLog(0, "Auto detection service log\n");


        // Check for root privileges before launching
        bool notAdmin = elevated();
        if (notAdmin) {
            Log::Logger::getInstance()->info(
                    "Program is not run as root. This is required to use the auto connect feature");
            AddLog(2, "Admin privileges is required to run the Auto-Connect feature");

            return;
        }

        Log::Logger::getInstance()->info("Start looking for ethernet adapters");

        skipUserFriendlySleepDelay = true;
        std::vector<AutoConnect::Result> res = autoConnect.findEthernetAdapters(false, false);
        skipUserFriendlySleepDelay = false;

        bool foundSupportedAdapter = false;
        bool foundSearchedAdapter = false;
        for (const auto &r: res) {
            if (r.supports)
                foundSupportedAdapter = true;

            if (r.searched)
                foundSearchedAdapter = true;
        }

        if (foundSupportedAdapter)
            autoConnect.start(res);
        else
            AddLog(2, "Did not find supported network adapters");

        // Clear list if we suddenly dont find supported adapters or searched adapters.
        if (!foundSearchedAdapter && !foundSupportedAdapter) {
            entryConnectDeviceList.clear();
            refreshAdapterList = true;
        }

    }

    void createDefaultElement(VkRender::GuiObjectHandles *handles, const VkRender::EntryConnectDevice &entry) {
        VkRender::Device el;

        el.name = entry.profileName;
        el.IP = entry.IP;
        el.state = AR_STATE_JUST_ADDED;
        el.cameraName = entry.cameraName;
        el.interfaceName = entry.interfaceName;
        el.interfaceDescription = entry.description;
        el.clicked = true;
        el.interfaceIndex = entry.interfaceIndex;
        el.baseUnit = handles->nextIsRemoteHead ? CRL_BASE_REMOTE_HEAD : CRL_BASE_MULTISENSE;

        handles->devices.emplace_back(el);

        Log::Logger::getInstance()->info("Connect clicked for Default Device");
        Log::Logger::getInstance()->info("Using: Ip: {}, and profile: {} for {}", entry.IP, entry.profileName,
                                         entry.description);
    }


    void sidebarElements(VkRender::GuiObjectHandles *handles) {
        auto& devices = handles->devices;
        for (int i = 0; i < devices.size(); ++i) {
            auto &e = devices.at(i);
            std::string buttonIdentifier;
            // Set colors based on state
            switch (e.state) {
                case AR_STATE_CONNECTED:
                    break;
                case AR_STATE_CONNECTING:
                    buttonIdentifier = "Connecting";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLBlueIsh);
                    break;
                case AR_STATE_ACTIVE:
                    buttonIdentifier = "Active";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray421);
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.42f, 0.31f, 1.0f));
                    break;
                case AR_STATE_INACTIVE:
                    buttonIdentifier = "Inactive";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);
                    break;
                case AR_STATE_DISCONNECTED:
                    buttonIdentifier = "Disconnected";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLRed);
                    break;
                case AR_STATE_UNAVAILABLE:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::CRLDarkGray425);
                    buttonIdentifier = "Unavailable";
                    break;
                case AR_STATE_JUST_ADDED:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    buttonIdentifier = "Added...";
                    break;
                case AR_STATE_DISCONNECT_AND_FORGET:
                case AR_STATE_REMOVE_FROM_LIST:
                case AR_STATE_RESET:
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
            if (ImGui::SmallButton("X")) {
                // delete and disconnect devices
                handles->devices.at(i).state = AR_STATE_DISCONNECT_AND_FORGET;
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(3);
                ImGui::EndChild();
                continue;
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

            buttonIdentifier += "##" + e.IP;
            e.clicked = ImGui::Button(buttonIdentifier.c_str(),
                                      ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2));

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
    }

    void addPopup(VkRender::GuiObjectHandles *handles) {
        ImGui::SetNextWindowSize(ImVec2(handles->info->popupWidth, handles->info->popupHeight), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);


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
            std::string title = "Connect to VkRender";
            ImVec2 size = ImGui::CalcTextSize(title.c_str());
            float anchorPoint = (handles->info->popupWidth - size.x) / 2; // Make a title in center of popup window


            ImGui::Dummy(ImVec2(0.0f, size.y));

            ImGui::SetCursorPosX(anchorPoint);
            ImGui::Text("%s", title.c_str());
            ImGui::PopFont();
            //ImGui::Separator();
            //ImGui::InputText("Profile name##1", inputName.data(),inputFieldNameLength);

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
            ImGui::Funcs::MyInputText("##inputProfileName", &entry.profileName);
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
            static bool active = false;

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

            ImGui::Dummy(ImVec2(0.0f, 25.0f));
            ImGui::PopStyleColor(); // ImGuiCol_FrameBg
            ImGui::PopStyleVar(2); // RadioButton

            autoDetectCamera();

            /** AUTOCONNECT FIELD BEGINS HERE*/
            if (connectMethodSelector == AUTO_CONNECT) {
                entry.cameraName = "AutoConnect";

                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Text("Status:");
                if (dontRunAutoConnect) {
                    ImGui::SameLine(0, 15.0f);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));

                    if (ImGui::SmallButton("Restart?")) {
                        dontRunAutoConnect = false;
                        entryConnectDeviceList.clear();
                        autoConnect.clearSearchedAdapters();
                        entry.reset();
                    }
                    ImGui::PopStyleColor();
                }
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 20.0f);
                /** STATUS SPINNER */
                // Create child window regardless of gif spinner state in order to keep cursor position constant
                ImGui::BeginChild("Gif viewer", ImVec2(40.0f, 40.0f), false, ImGuiWindowFlags_NoDecoration);
                if (autoConnect.running)
                addSpinnerGif(handles);
                ImGui::EndChild();

                ImGui::SameLine(0, 250.0f);
                ImGui::HelpMarker(" If no packet at adapter is received try the following: \n "
                                  " 1. Reconnect ethernet cables \n "
                                  " 2. Power cycle the camera \n "
                                  " 3. Wait 20-30 seconds. If no packet is received contact support  \n\n");


                ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::CRLDarkGray425);
                const char *id = "Log Window";
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
                        const char *line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1)
                                                                                : buf_end;
                        // Weird index magic. colors is an ImGui vector.
                        if (line_no > 0 && line_no < colors.size() - 1) {
                            int col = colors[line_no + 1];
                            switch (col) {
                                case 0:
                                    lastLogTextColor = VkRender::TextColorGray;
                                    break;
                                case 1:
                                    lastLogTextColor = VkRender::TextGreenColor;
                                    break;
                                case 2:
                                    lastLogTextColor = VkRender::TextRedColor;
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

                if (time_span.count() > 0.35 && autoConnect.running) {
                    searchingTextAnimTimer = std::chrono::steady_clock::now();
                    dots.append(".");

                    if (dots.size() > 4)
                        dots.clear();

                } else if (!autoConnect.running)
                    dots = ".";

                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);

                if (entryConnectDeviceList.empty() && autoConnect.running) {
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
                ImGui::BeginChild("##ResultsChild", ImVec2(handles->info->popupWidth - (20.0f * 2.0f), 50.0f), true, 0);
                for (int n = 0; n < entryConnectDeviceList.size(); n++) {

                    if (ImGui::Selectable(entryConnectDeviceList[n].cameraName.c_str(), resultsComboIndex == n,
                                          ImGuiSelectableFlags_DontClosePopups,
                                          ImVec2(handles->info->popupWidth - (20.0f * 2), 15.0f))) {

                        resultsComboIndex = n;
                        entryConnectDeviceList[n].profileName = entryConnectDeviceList[n].cameraName; // Keep profile name if user inputted this before auto-connect is finished
                        entry = entryConnectDeviceList[n];
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
                ImGui::Checkbox(" Remote Head", &handles->nextIsRemoteHead);
                ImGui::PopStyleColor();

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
                ImGui::Funcs::MyInputText("##inputIP", &entry.IP);
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
                    adapters = autoConnect.findEthernetAdapters(false,
                                                                true); // Don't log it but dont ignore searched adapters
                    interfaceDescriptionList.clear();
                    indexList.clear();
                }

                // immediate mode vector item ordering
                // -- requirements --
                // Always have a base item in it.
                // if we push back other items then remove base item
                // No identical items
                for (const auto &a: adapters) {
                    if (a.supports && !Utils::isInVector(interfaceDescriptionList, a.description)) {
                        interfaceDescriptionList.push_back(a.description);
                        indexList.push_back(a.index);

                        if (Utils::isInVector(interfaceDescriptionList, "No adapters found")) {
                            Utils::delFromVector(interfaceDescriptionList, "No adapters found");
                            auto newEndIterator = std::remove(indexList.begin(), indexList.end(), 0);
                        }
                    }
                }


                if (!Utils::isInVector(interfaceDescriptionList, "No adapters found") &&
                    interfaceDescriptionList.empty()) {
                    interfaceDescriptionList.emplace_back("No adapters found");
                    indexList.push_back(0);
                }


                entry.description = interfaceDescriptionList[ethernetComboIndex];  // Pass in the preview value visible before opening the combo (it could be anything)
                entry.interfaceIndex = indexList[ethernetComboIndex];
                if (!adapters.empty())
                    entry.interfaceName = adapters[ethernetComboIndex].networkAdapter;
                static ImGuiComboFlags flags = 0;
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                if (ImGui::BeginCombo("##SelectAdapter", entry.description.c_str(), flags)) {
                    for (int n = 0; n < interfaceDescriptionList.size(); n++) {
                        const bool is_selected = (ethernetComboIndex == n);
                        if (ImGui::Selectable(interfaceDescriptionList[n].c_str(), is_selected))
                            ethernetComboIndex = n;

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopStyleColor(); // ImGuiCol_FrameBg
                entry.cameraName = "Manual";

                ImGui::Dummy(ImVec2(40.0f, 10.0));
                ImGui::Dummy(ImVec2(20.0f, 0.0));
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLTextGray);
                ImGui::Checkbox("  Configure System Network", &handles->configureNetwork);
                ImGui::SameLine(0, 20.0f);
                ImGui::Checkbox("  Remote Head", &handles->nextIsRemoteHead);

                if (handles->configureNetwork) {
                    if (elevated()) {
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::CRLRed);
                        ImGui::Dummy(ImVec2(40.0f, 10.0));
                        ImGui::Dummy(ImVec2(20.0f, 0.0));
                        ImGui::SameLine();
                        ImGui::Text("Launch app as admin to \nconfigure system network");
                        enableConnectButton = false;
                        ImGui::PopStyleColor();
                    }

                } else
                    enableConnectButton = true;
                ImGui::PopStyleColor();

            } /** VIRTUAL_CONNECT FIELD BEGINS HERE*/
            else if (connectMethodSelector == VIRTUAL_CONNECT) {
                entry.profileName = "Virtual Camera";
                entry.interfaceName = "lol";
                entry.cameraName = "Virtual Camera";
            }


            /** CANCEL/CONNECT FIELD BEGINS HERE*/
            ImGui::Dummy(ImVec2(0.0f, 40.0f));
            ImGui::SetCursorPos(ImVec2(0.0f, handles->info->popupHeight - 50.0f));
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            bool btnCancel = ImGui::Button("cancel", ImVec2(150.0f, 30.0f));
            ImGui::SameLine(0, 110.0f);
            if (!entry.ready(handles->devices, entry) || !enableConnectButton) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::TextColorGray);
            }
            btnConnect = ImGui::Button("connect", ImVec2(150.0f, 30.0f));
            if (!entry.ready(handles->devices, entry) || !enableConnectButton) {
                ImGui::PopStyleColor();
                ImGui::PopItemFlag();
            }
            ImGui::Dummy(ImVec2(0.0f, 5.0f));

            if (btnCancel) {
                ImGui::CloseCurrentPopup();
                autoConnect.stop();
            }

            if (btnConnect && entry.ready(handles->devices, entry) && enableConnectButton) {
                createDefaultElement(handles, entry);
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

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
