#ifndef SIDEBAR_H
#define SIDEBAR_H


#include "Viewer/VkRender/Editors/MultiSenseViewer/Modules/LibMultiSense/CommonHeader.h"



#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"


namespace VkRender {

    class SideBarLayer : public Layer {


    static void addDeviceButton(GuiObjectHandles& uiContext) {
        ImGui::SetCursorPos(ImVec2(0.0f, uiContext.editorUi->height - 50.0f));

        ImGui::PushStyleColor(ImGuiCol_Button, Colors::CRLBlueIsh);
        if (ImGui::Button("ADD DEVICE", ImVec2(uiContext.editorUi->width, 35.0f))) {
            uiContext.shared->openAddDevicePopup = !uiContext.shared->openAddDevicePopup;

            uiContext.usageMonitor->userClickAction("ADD_DEVICE", "button", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::PopStyleColor();
    }


    void drawProfilesInSidebar(GuiObjectHandles& uiContext) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLDarkGray425);
        std::default_random_engine rng;
        float sidebarElementHeight = 140.0f;
        /*
        for (auto &profile: uiContext.shared->multiSenseRendererBridge->getProfileList()) {

            // Color the sidebar and window depending on the connection state. Must be found before we start drawing the window containing the profile.
            std::string buttonIdentifier = "InvalidConnectionState";
            switch (profile.connectionState) {
                case MultiSense::MULTISENSE_DISCONNECTED:
                    buttonIdentifier = "Disconnected";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, Colors::CRLRed);
                    break;
                case MultiSense::MULTISENSE_UNAVAILABLE:
                    buttonIdentifier = "Unavailable";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, Colors::CRLDarkGray425);
                    break;

                case MultiSense::MULTISENSE_CONNECTED:
                    buttonIdentifier = "Active";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray421);
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.42f, 0.31f, 1.0f));
                    break;

                case MultiSense::MULTISENSE_CHANNEL_BUSY:
                    buttonIdentifier = "Working...";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424);
                    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
                    break;
            }
            // Connect button
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            std::string winId = profile.createInfo.profileName + "Child" + std::to_string(rng());;
            ImGui::BeginChild(winId.c_str(), ImVec2(uiContext.editorUi->width, sidebarElementHeight),
                              false, ImGuiWindowFlags_NoDecoration);

            ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
            bool removeProfileButtonX = ImGui::SmallButton("X");
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
                ImGui::PushFont(uiContext.info->font24);
                lineSize = ImGui::CalcTextSize(profile.createInfo.profileName.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(cursorPos);
                ImGui::Text("%s", profile.createInfo.profileName.c_str());
                ImGui::PopFont();
            }
            // Camera Name
            {
                ImGui::PushFont(uiContext.info->font13);
                lineSize = ImGui::CalcTextSize(profile.createInfo.cameraModel.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
                ImGui::Text("%s", profile.createInfo.cameraModel.c_str());
                ImGui::PopFont();
            }
            // Camera IP Address
            {
                ImGui::PushFont(uiContext.info->font13);
                lineSize = ImGui::CalcTextSize(profile.createInfo.inputIP.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));

                ImGui::Text("%s", profile.createInfo.inputIP.c_str());
                ImGui::PopFont();
            }

            // Status Button
            {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::PushFont(uiContext.info->font18);
                //ImGuiStyle style = ImGui::GetStyle();
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
                cursorPos.x = window_center.x - (ImGui::GetFontSize() * 10 / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
            }

            ImVec2 uv0 = ImVec2(0.0f, 0.0f); // UV coordinates for lower-left
            ImVec2 uv1 = ImVec2(1.0f, 1.0f);
            ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 0.3f); // No tint


            bool deviceButton;
            static size_t gifFrameIndex = 0;
            if (profile.connectionState == MultiSense::MULTISENSE_CHANNEL_BUSY) {
                deviceButton = ImGui::ButtonWithGif(buttonIdentifier.c_str(), ImVec2(ImGui::GetFontSize() * 10, 35.0f),
                                                    uiContext.info->gif.image[gifFrameIndex], ImVec2(35.0f, 35.0f),
                                                    uv0,
                                                    uv1,
                                                    tint_col, VkRender::Colors::CRLBlueIsh);
            } else {
                deviceButton = ImGui::Button(buttonIdentifier.c_str(),
                                             ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2));
            }

            ImGui::PopStyleColor(2);

            auto now = std::chrono::system_clock::now();
            std::chrono::duration<float, std::milli> elapsed_milliseconds = now - uiContext.info->gif.lastUpdateTime;

            if (elapsed_milliseconds.count() >= 33.0) {
                gifFrameIndex++;
                uiContext.info->gif.lastUpdateTime = now;
            }

            if (gifFrameIndex >= uiContext.info->gif.totalFrames)
                gifFrameIndex = 0;

            ImGui::PopFont();
            ImGui::PopStyleVar(2);

            if (deviceButton && profile.connectionState != MultiSense::MULTISENSE_CONNECTED) {
                uiContext.shared->multiSenseRendererBridge->connect(profile);
            } else if (deviceButton) {
                uiContext.shared->multiSenseRendererBridge->disconnect(profile);
            }

            if (removeProfileButtonX) {
                uiContext.shared->multiSenseRendererBridge->removeProfile(profile);
            }

            ImGui::EndChild();
        }
            */
        ImGui::PopStyleColor();
    }

    void drawSideBar(GuiObjectHandles& uiContext) {
        ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));

        ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main);
        // Begin the sidebar as a child window
        ImGui::BeginChild("Sidebar", ImVec2(uiContext.editorUi->width, uiContext.editorUi->height), false,
                          ImGuiWindowFlags_NoScrollWithMouse);
        //addPopup(uiContext);
        //askUsageLoggingPermissionPopUp(uiContext);
        // Settings button

        {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            if (ImGui::Button("Settings", ImVec2(uiContext.editorUi->width, 17.0f))) {
                uiContext.showDebugWindow = !uiContext.showDebugWindow;
                uiContext.usageMonitor->userClickAction("Settings", "button", ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopStyleVar();
        }



        drawProfilesInSidebar(uiContext);
        addDeviceButton(uiContext);

        // Add version number
        ImGui::SetCursorPos(ImVec2(0.0f, uiContext.info->editorHeight - 10.0f));
        ImGui::PushFont(uiContext.info->font8);
        ImGui::Text("%s", (std::string("Ver: ") + ApplicationConfig::getInstance().getAppVersion()).c_str());
        ImGui::PopFont();

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }





    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {
            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground;
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(handles.editorUi->width, handles.editorUi->height));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("WelcomeScreen", &pOpen, window_flags);
            ImVec2 winSize = ImGui::GetWindowSize();

            // Keep all logic inside each function call. Call each element here for a nice orderly overview
            drawSideBar(handles);
            // Draw welcome screen
            //drawWelcomeScreen(uiContext);
            // Draw Main preview page, after we obtained a successfull connection
            //drawConfigurationPage(uiContext);
            // Draw the 2D view (Camera stream, layouts etc..)
            //drawCameraStreamView(uiContext);
            // Draw the 3D View

            ImGui::End();
            ImGui::PopStyleVar();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}

#endif //SIDEBAR_H
