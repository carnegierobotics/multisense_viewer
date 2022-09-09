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

#include "AutoConnectLinux.h"

#define AutoConnectHandle AutoConnectLinux
#define elevated() getuid()
#endif

#include <MultiSense/src/tools/Utils.h>

#include <algorithm>
#include <queue>
#include "imgui_internal.h"
#include "imgui.h"
#include "Layer.h"
#include "imgui_user.h"
#include <unistd.h>
#include <sys/types.h>

class SideBar : public AR::Layer {
public:

    // Create global object for convenience in other functions
    AR::GuiObjectHandles *handles;
    AutoConnectHandle autoConnect{};
    bool refreshAdapterList = true; // Set to true to find adapters on next call
    std::vector<AutoConnect::AdapterSupportResult> adapters;
    std::vector<std::string> interfaceNameList;
    std::vector<uint32_t> indexList;

    uint32_t gifFrameIndex = 0;
    std::chrono::steady_clock::time_point gifFrameTimer;

    void OnAttach() override {
        autoConnect.setDetectedCallback(SideBar::onCameraDetected, this);
        autoConnect.setEventCallback(SideBar::onEvent);
        gifFrameTimer = std::chrono::steady_clock::now();
    }

    void onFinishedRender() override {

    }


    void OnUIRender(AR::GuiObjectHandles *_handles) override {
        this->handles = _handles;
        AR::GuiLayerUpdateInfo *info = handles->info;


        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth, info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.031, 0.078, 0.129, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("SideBar", &pOpen, window_flags);


        // Docking
        auto *wnd = ImGui::FindWindowByName("GUI");
        /*
        if (wnd) {
            ImGuiDockNode *node = wnd->DockNode;
            if (node)
                node->WantHiddenTabBarToggle = true;

        }
         */

        ImGui::TextUnformatted(info->title.c_str());
        ImGui::TextUnformatted(info->deviceName.c_str());

        // Update frame time display
        if (info->firstFrame) {
            std::rotate(info->frameTimes.begin(), info->frameTimes.begin() + 1,
                        info->frameTimes.end());
            float frameTime = 1000.0f / (info->frameTimer * 1000.0f);
            info->frameTimes.back() = frameTime;
            if (frameTime < info->frameTimeMin) {
                info->frameTimeMin = frameTime;
            }
            if (frameTime > info->frameTimeMax) {
                info->frameTimeMax = frameTime;
            }
        }

        ImGui::PlotLines("Frame Times", &info->frameTimes[0], 50, 0, "", info->frameTimeMin,
                         info->frameTimeMax, ImVec2(0, 80));

        ImGui::SetNextWindowSize(ImVec2(info->popupWidth, info->popupHeight), ImGuiCond_Always);

        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, AR::PopupBackground);


        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

        if (ImGui::BeginPopupModal("add_device_modal", nullptr,
                                   ImGuiWindowFlags_NoTitleBar)) {


            ImGui::PushFont(handles->info->font24);
            std::string title = "Connect to MultiSense";
            ImVec2 size = ImGui::CalcTextSize(title.c_str());
            float anchorPoint = (handles->info->popupWidth - size.x) / 2; // Make a title in center of popup window


            ImGui::Dummy(ImVec2(0.0f, size.y));

            ImGui::SetCursorPosX(anchorPoint);
            ImGui::Text("%s", title.c_str());
            ImGui::PopFont();
            //ImGui::Separator();
            //ImGui::InputText("Profile name##1", inputName.data(),inputFieldNameLength);

            ImGui::Dummy(ImVec2(0.0f, 25.0f));

            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::Text("Profile Name:");
            // Create input field with resizable data structure
            {
                // To wire InputText() with std::string or any other custom string type,
                // you can use the ImGuiInputTextFlags_CallbackResize flag + create a custom ImGui::InputText() wrapper
                // using your preferred type. See misc/cpp/imgui_stdlib.h for an implementation of this using std::string.
                struct Funcs {
                    static int MyResizeCallback(ImGuiInputTextCallbackData *data) {
                        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                            auto *my_str = (std::string *) data->UserData;
                            IM_ASSERT(my_str->data() == data->Buf);
                            my_str->resize(
                                    data->BufSize); // NB: On resizing calls, generally data->BufSize == data->BufTextLen + 1
                            data->Buf = my_str->data();
                        }
                        return 0;
                    }

                    // Note: Because ImGui:: is a namespace you would typically add your own function into the namespace.
                    // For example, you code may declare a function 'ImGui::InputText(const char* label, MyString* my_str)'
                    static bool
                    MyInputText(const char *label, std::string *my_str, ImGuiInputTextFlags flags = 0) {
                        IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
                        return ImGui::InputText(label, my_str->data(), (size_t) my_str->size(),
                                                flags | ImGuiInputTextFlags_CallbackResize |
                                                ImGuiInputTextFlags_AutoSelectAll,
                                                Funcs::MyResizeCallback, (void *) my_str);
                    }
                };

                // Note that because we need to store a terminating zero character, our size/capacity are 1 more
                // than usually reported by a typical string class.
                //if (inputName.empty())
                //    inputName.push_back(0);

                ImGui::PushStyleColor(ImGuiCol_FrameBg, AR::PopupTextInputBackground);
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                Funcs::MyInputText("##inputProfileName", &inputName);
                ImGui::Dummy(ImVec2(0.0f, 30.0f));
            }


            {
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::Text("Find Camera:");
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
            }

            {
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 0.0f));
                ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(10.0f, 0.0f));

                ImGui::RadioButton("Manual", &connectMethodSelector, 1);
                ImGui::SameLine();
                ImGui::RadioButton("Automatic", &connectMethodSelector, 2);
                ImGui::SameLine();
                ImGui::RadioButton("Use Virtual", &connectMethodSelector, 0);
                /*
                                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
                autoOrManual |= ImGui::Button("Automatic", ImVec2(150.0f, 35.0f));
                ImGui::SameLine(0, 10.0f);
                if (ImGui::Button("Manual", ImVec2(150.0f, 35.0f))){
                    AddLog("[%05d] Hello, current time is %.1f\n",
                           ImGui::GetFrameCount(), ImGui::GetTime());
                }
                ImGui::PopStyleVar();
                */
                ImGui::Dummy(ImVec2(0.0f, 25.0f));

                ImGui::PopStyleColor(); // ImGuiCol_FrameBg
                ImGui::PopStyleVar(2); // RadioButton

            }

            if (connectMethodSelector == AUTO_CONNECT) {

                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();
                if (ImGui::Button("Start", ImVec2(40.0f, 0.0f))) {
                    autoDetectCamera();
                }
                ImGui::SameLine(0, 350.0f);
                ImGui::HelpMarker(" If no packet at adapter is received try the following: \n "
                                  " 1. Reconnect ethernet cables \n "
                                  " 2. Power cycle the camera \n "
                                  " 3. Wait 20-30 seconds. If no packet is received contact support  \n\n");


                ImGui::PushStyleColor(ImGuiCol_ChildBg, AR::PopupTextInputBackground);
                const char *id = "Log Window";
                ImGui::Dummy(ImVec2(20.0f, 0.0f));

                ImGui::SameLine();
                ImGui::BeginChild(id, ImVec2(handles->info->popupWidth - 40.0f, 85.0f), false, 0);

                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0));

                const char *buf = Buf.begin();
                const char *buf_end = Buf.end();
                // The simplest and easy way to display the entire buffer:
                //   ImGui::TextUnformatted(buf_begin, buf_end);
                // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward
                // to skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
                // within the visible area.
                // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them
                // on your side is recommended. Using ImGuiListClipper requires
                // - A) random access into your data
                // - B) items all being the  same height,
                // both of which we can handle since we an array pointing to the beginning of each line of text.
                // When using the filter (in the block of code above) we don't have random access into the data to display
                // anymore, which is why we don't use the clipper. Storing or skimming through the search result would make
                // it possible (and would be recommended if you want to search through tens of thousands of entries).
                bool colored = false;


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
                                    lastLogTextColor = AR::TextColorGray;
                                    break;
                                case 1:
                                    lastLogTextColor = AR::TextGreenColor;
                                    break;
                                case 2:
                                    lastLogTextColor = AR::TextRedColor;
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

                ImVec2 size = ImVec2(124.0f,
                                     124.0f);                     // TODO dont make use of these hardcoded values. Use whatever values that were gathered during texture initialization
                ImVec2 uv0 = ImVec2(0.0f, 0.0f);                        // UV coordinates for lower-left
                ImVec2 uv1 = ImVec2(1.0f, 1.0f);

                ImVec4 bg_col = ImVec4(0.054, 0.137, 0.231, 1.0f);         // Match bg color
                ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);       // No tint

                ImGui::Image(handles->info->gif.image[gifFrameIndex], size, uv0, uv1, bg_col, tint_col);

                auto time = std::chrono::steady_clock::now();
                std::chrono::duration<float> time_span =
                        std::chrono::duration_cast<std::chrono::duration<float>>(time - gifFrameTimer);

                if (time_span.count() >  ((float) *handles->info->gif.delay) / 1000.0f) {
                    gifFrameTimer = std::chrono::steady_clock::now();
                    gifFrameIndex++;
                }

                if (gifFrameIndex == handles->info->gif.totalFrames)
                    gifFrameIndex = 0;

                ImGui::EndChild();
                cameraName = "Automatic Entry";
            } else if (connectMethodSelector == MANUAL_CONNECT) {
                {
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::Text("Camera IP:");
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                }
                // To wire InputText() with std::string or any other custom string type,
                // you can use the ImGuiInputTextFlags_CallbackResize flag + create a custom ImGui::InputText() wrapper
                // using your preferred type. See misc/cpp/imgui_stdlib.h for an implementation of this using std::string.
                struct Funcs {
                    static int MyResizeCallback(ImGuiInputTextCallbackData *data) {
                        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                            auto *my_str = (std::string *) data->UserData;
                            IM_ASSERT(my_str->data() == data->Buf);
                            my_str->resize(
                                    data->BufSize); // NB: On resizing calls, generally data->BufSize == data->BufTextLen + 1
                            data->Buf = my_str->data();
                        }
                        return 0;
                    }

                    // Note: Because ImGui:: is a namespace you would typically add your own function into the namespace.
                    // For example, you code may declare a function 'ImGui::InputText(const char* label, MyString* my_str)'
                    static bool
                    MyInputText(const char *label, std::string *my_str, ImGuiInputTextFlags flags = 0) {
                        IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
                        return ImGui::InputText(label, my_str->data(), (size_t) my_str->size(),
                                                flags | ImGuiInputTextFlags_CallbackResize,
                                                Funcs::MyResizeCallback, (void *) my_str);
                    }
                };

                // Note that because we need to store a terminating zero character, our size/capacity are 1 more
                // than usually reported by a typical string class.


                ImGui::PushStyleColor(ImGuiCol_FrameBg, AR::PopupTextInputBackground);
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                Funcs::MyInputText("##inputIP", &inputIP);
                ImGui::Dummy(ImVec2(0.0f, 10.0f));

                {
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine(0.0f, 10.0f);
                    ImGui::Text("Select network adapter:");
                    ImGui::SameLine(0.0f, 10.0f);
                }

                if (ImGui::SmallButton("Refresh list") || refreshAdapterList) {
                    skipUserFriendlySleepDelay = true;
                    adapters = autoConnect.findEthernetAdapters(false);
                    skipUserFriendlySleepDelay = false;
                    refreshAdapterList = false;

                    // Reset list of adapters
                    interfaceNameList.clear();
                }
                ImGui::Dummy(ImVec2(0.0f, 5.0f));


                // immediate mode vector item ordering
                // -- requirements --
                // Always have a base item in it.
                // if push back other items then remove base item
                // No identical items

                for (const auto &a: adapters) {


                    if (a.supports && !Utils::isInVector(interfaceNameList, a.description)) {
                        interfaceNameList.push_back(a.description);
                        indexList.push_back(a.index);
                        if (Utils::isInVector(interfaceNameList, "No adapters found"))
                            Utils::delFromVector(interfaceNameList, "No adapters found");
                    }
                }

                if (!Utils::isInVector(interfaceNameList, "No adapters found") && interfaceNameList.empty())
                    interfaceNameList.emplace_back("No adapters found");


                interfaceName = interfaceNameList[presetItemIdIndex];  // Pass in the preview value visible before opening the combo (it could be anything)
                interfaceIndex = indexList[presetItemIdIndex];
                static ImGuiComboFlags flags = 0;
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(handles->info->popupWidth - 40.0f);
                if (ImGui::BeginCombo("##SelectAdapter", interfaceName.c_str(), flags)) {
                    for (int n = 0; n < interfaceNameList.size(); n++) {
                        const bool is_selected = (presetItemIdIndex == n);
                        if (ImGui::Selectable(interfaceNameList[n].c_str(), is_selected))
                            presetItemIdIndex = n;

                        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopStyleColor(); // ImGuiCol_FrameBg
                cameraName = "Manual Entry";

            } else if (connectMethodSelector == VIRTUAL_CONNECT) {
                inputName = "Virtual Camera";
                interfaceName = "lol";
                cameraName = "Virtual Camera";
                inputIP = "Local";
            }
            ImGui::Dummy(ImVec2(0.0f, 40.0f));


            ImGui::SetCursorPos(ImVec2(0.0f, handles->info->popupHeight - 50.0f));
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            bool btnCancel = ImGui::Button("cancel", ImVec2(150.0f, 30.0f));
            ImGui::SameLine(0, 110.0f);
            btnConnect = ImGui::Button("connect", ImVec2(150.0f, 30.0f));
            ImGui::Dummy(ImVec2(0.0f, 5.0f));

            if (btnCancel) {
                ImGui::CloseCurrentPopup();
                autoConnect.stop();
            }

            if (btnConnect) {
                bool ipAlreadyInUse = false;
                bool profileNameTaken = false;
                bool virtualCameraTaken = false;
                bool profileNameEmpty = false;
                bool IpEmptyOrLocal = false;

                if (inputName.empty())
                    profileNameEmpty = true;

                // Loop through devices and check that it doesn't exist already.
                for (auto &d: devices) {
                    if (d.IP == inputIP) {
                        ipAlreadyInUse = true;
                        Log::Logger::getInstance()->info("Ip in use");

                    }
                    if (d.name == inputName) {
                        profileNameTaken = true;
                        Log::Logger::getInstance()->info("Profile name not valid");
                    }

                    if (d.name == "Virtual Camera" && inputName == "Virtual Camera") {
                        virtualCameraTaken = true;
                        Log::Logger::getInstance()->info("Virtual camera occupied");
                    }
                }

                if (!ipAlreadyInUse && !profileNameTaken && !virtualCameraTaken && !profileNameEmpty) {
                    createDefaultElement(inputName.data(), inputIP.data(), interfaceName.data(), interfaceIndex, cameraName.data());

                    inputName = "";
                    inputIP = "";
                    interfaceName = "";
                    ImGui::CloseCurrentPopup();
                } else {

                }


            }


            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(4); // popup style vars
        ImGui::PopStyleColor(); // popup bg color

        ImGui::Spacing();
        ImGui::Spacing();

        if (!devices.empty())
            sidebarElements();

        addDeviceButton(handles);
        ImGui::End();
        ImGui::PopStyleColor(); // bg color
        ImGui::PopStyleVar();
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
    std::vector<AR::Element> devices;

    bool btnConnect = false;
    bool btnAdd = false;
    bool skipUserFriendlySleepDelay = false;

    enum {
        VIRTUAL_CONNECT = 0,
        MANUAL_CONNECT = 1,
        AUTO_CONNECT = 2
    };
    int connectMethodSelector = 1;
    ImGuiTextBuffer Buf;
    ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
    ImVector<int> colors;
    ImVec4 lastLogTextColor = AR::TextColorGray;

    uint32_t presetItemIdIndex = 0;
    std::string inputIP = "10.66.176.21";
    std::string inputName = "MultiSense";
    std::string interfaceName;
    std::string cameraName;
    uint32_t interfaceIndex;


    static void onEvent(std::string event, void *ctx, int color = 0) {
        auto *app = static_cast<SideBar *>(ctx);

        // added a delay for user-friendliness. Also works great cause switching colors should be done on main thread
        // but Push/Pop style works here because of the delay.
        if (!app->skipUserFriendlySleepDelay) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        app->AddLog(color, "[INFO] %s\n", event.c_str());


    }

    static void onCameraDetected(AutoConnect::Result res, void *ctx) {
        auto *app = static_cast<SideBar *>(ctx);


        crl::multisense::Channel *ptr = app->autoConnect.getCameraChannel();
        crl::multisense::system::DeviceInfo info;
        ptr->getDeviceInfo(info);
        Log::Logger::getInstance()->info(
                "AUTOCONNECT: Found Camera on IP: {}, using Adapter: {}, adapter long name: {}, Camera returned name {}",
                res.cameraIpv4Address.c_str(), res.networkAdapter.c_str(), res.networkAdapterLongName.c_str(),
                info.name.c_str());

        app->inputIP = res.cameraIpv4Address;
        app->inputName = info.name;
        app->interfaceName = res.networkAdapter;
        app->interfaceIndex = res.index;
        app->cameraName = info.name;

        if (app->inputName == "Multisense S30") // TODO: Avoid hardcoded if-cond here
            app->presetItemIdIndex = 1;
        else if (app->inputName == "MultiSense S21")
            app->presetItemIdIndex = 2;
        else
            app->presetItemIdIndex = 0;

        app->autoConnect.setProgramClose(true);

    }

    void autoDetectCamera() {
        // If already running just check if we should close it before opening it again.
        if (autoConnect.shouldProgramClose())
            autoConnect.stop();

        Buf.clear();
        LineOffsets.clear();
        LineOffsets.push_back(0);
        colors.clear();
        colors.push_back(0);

        AddLog(0, "Auto detection service log\n");


        // Check for root privileges before launching
        bool admin = elevated();
        if (admin) {
            Log::Logger::getInstance()->info(
                    "Program is not run as root. This is required to use the auto connect feature");
            AddLog(2, "Admin privileges is required to run the Auto-Connect feature");

            return;
        }

        Log::Logger::getInstance()->info("Start looking for ethernet adapters");

        skipUserFriendlySleepDelay = true;
        std::vector<AutoConnect::AdapterSupportResult> res = autoConnect.findEthernetAdapters(false);
        skipUserFriendlySleepDelay = false;

        bool foundSupportedAdapter = false;
        for (const auto &r: res) {
            if (r.supports)
                foundSupportedAdapter = true;
        }

        if (foundSupportedAdapter)
            autoConnect.start(res);
        else
            AddLog(2, "Did not find any supported network adapters");
    }


    void createDefaultElement(char *name, char *ip, char *interfaceName, uint32_t idx, const char *camName = "empty") {
        AR::Element el;

        el.name = name;
        el.IP = ip;
        el.state = AR_STATE_JUST_ADDED;
        el.cameraName = camName;
        el.interfaceName = interfaceName;
        el.clicked = true;
        el.interfaceIndex = idx;

        devices.emplace_back(el);

        handles->devices = &devices;

        Log::Logger::getInstance()->info("Connect clicked for Default Device");
        Log::Logger::getInstance()->info("Using: Ip: {}, and preset: {}", ip, name);
    }


    void sidebarElements() {
        for (int i = 0; i < devices.size(); ++i) {
            auto &e = devices[i];
            std::string buttonIdentifier;
            // Set colors based on state
            switch (e.state) {
                case AR_STATE_CONNECTED:
                    break;
                case AR_STATE_CONNECTING:
                    buttonIdentifier = "Connecting";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.19f, 0.33f, 0.48f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.98f, 0.65f, 0.00f, 1.0f));
                    break;
                case AR_STATE_ACTIVE:
                    buttonIdentifier = "Active";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.19f, 0.33f, 0.48f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.42f, 0.31f, 1.0f));
                    break;
                case AR_STATE_INACTIVE:
                    buttonIdentifier = "Inactive";
                    break;
                case AR_STATE_DISCONNECTED:
                    buttonIdentifier = "Disconnected";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.713f, 0.065f, 0.066f, 1.0f));
                    break;
                case AR_STATE_UNAVAILABLE:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    buttonIdentifier = "Unavailable";
                    break;
                case AR_STATE_JUST_ADDED:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    buttonIdentifier = "Added...";
                    break;
                case AR_STATE_RESET:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    break;
            }

            ImGui::SetCursorPos(ImVec2(5.0f, ImGui::GetCursorPosY()));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            std::string winId = e.name + "Child";
            ImGui::BeginChild(winId.c_str(), ImVec2(handles->info->sidebarWidth, handles->info->elementHeight),
                              false, ImGuiWindowFlags_NoDecoration);


            if (ImGui::SmallButton("X")) {
                devices.erase(devices.begin() + i);
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);
                ImGui::EndChild();
                continue;
            }
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
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();

            ImGui::EndChild();

            ImGui::PopStyleColor();
            ImGui::PopStyleVar();


        }
    }

    void addDeviceButton(AR::GuiObjectHandles *handles) {

        ImGui::SetCursorPos(ImVec2(handles->info->addDeviceLeftPadding,
                                   handles->info->height - handles->info->addDeviceBottomPadding));
        btnAdd = ImGui::Button("ADD DEVICE", ImVec2(handles->info->addDeviceWidth, handles->info->addDeviceHeight));

        if (btnAdd) {
            ImGui::OpenPopup("add_device_modal");

        }
    }


};


#endif //MULTISENSE_SIDEBAR_H
