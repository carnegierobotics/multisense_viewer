#ifndef SIDEBAR_H
#define SIDEBAR_H


#include "Viewer/Modules/MultiSense/CommonHeader.h"


#include "Viewer/Rendering/ImGui/Layer.h"
#include "Viewer/Rendering/ImGui/IconsFontAwesome6.h"


namespace VkRender {

    class SideBarLayer : public Layer {


        void addDeviceButton() {
            ImGui::SetCursorPos(ImVec2(0.0f, m_editor->ui()->height - 50.0f));

            ImGui::PushStyleColor(ImGuiCol_Button, Colors::CRLBlueIsh);
            if (ImGui::Button("ADD DEVICE", ImVec2(m_editor->ui()->width, 35.0f))) {
                //m_editor->ui()->shared->openAddDevicePopup = !m_editor->ui()->shared->openAddDevicePopup;

                m_context->usageMonitor()->userClickAction("ADD_DEVICE", "button", ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopStyleColor();
        }


        void drawProfilesInSidebar() {
            ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLDarkGray425);
            std::default_random_engine rng;
            float sidebarElementHeight = 140.0f;

            bool removeProfileButtonXClicked = false;
            std::shared_ptr<MultiSense::MultiSenseDevice> profileToRemove;

            for (auto &profile: m_context->multiSense()->getAllMultiSenseProfiles()) {
                // Color the sidebar and window depending on the connection state. Must be found before we start drawing the window containing the profile->
                std::string buttonIdentifier = "InvalidConnectionState";
                switch (profile->multiSenseTaskManager->connectionState()) {
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
                std::string winId = profile->profileCreateInfo.profileName + "Child" + std::to_string(rng());;
                ImGui::BeginChild(winId.c_str(), ImVec2(m_editor->ui()->width, sidebarElementHeight),
                                  false, ImGuiWindowFlags_NoDecoration);

                ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
                if (ImGui::SmallButton("X")) {
                    removeProfileButtonXClicked = true;
                    profileToRemove = profile;
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
                    ImGui::PushFont(m_editor->guiResources().font24);
                    lineSize = ImGui::CalcTextSize(profile->profileCreateInfo.profileName.c_str());
                    cursorPos.x = window_center.x - (lineSize.x / 2);
                    ImGui::SetCursorPos(cursorPos);
                    ImGui::Text("%s", profile->profileCreateInfo.profileName.c_str());
                    ImGui::PopFont();
                }
                // Camera Name
                {
                    ImGui::PushFont(m_editor->guiResources().font13);
                    lineSize = ImGui::CalcTextSize(profile->profileCreateInfo.cameraModel.c_str());
                    cursorPos.x = window_center.x - (lineSize.x / 2);
                    ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
                    ImGui::Text("%s", profile->profileCreateInfo.cameraModel.c_str());
                    ImGui::PopFont();
                }
                // Camera IP Address
                {
                    ImGui::PushFont(m_editor->guiResources().font13);
                    lineSize = ImGui::CalcTextSize(profile->profileCreateInfo.inputIP.c_str());
                    cursorPos.x = window_center.x - (lineSize.x / 2);
                    ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));

                    ImGui::Text("%s", profile->profileCreateInfo.inputIP.c_str());
                    ImGui::PopFont();
                }

                // Status Button
                {
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                    ImGui::PushFont(m_editor->guiResources().font18);
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
                if (profile->multiSenseTaskManager->connectionState() == MultiSense::MULTISENSE_CHANNEL_BUSY) {
                    deviceButton = ImGui::ButtonWithGif(buttonIdentifier.c_str(),
                                                        ImVec2(ImGui::GetFontSize() * 10, 35.0f),
                                                        m_editor->guiResources().gif.image[gifFrameIndex], ImVec2(35.0f, 35.0f),
                                                        uv0,
                                                        uv1,
                                                        tint_col, VkRender::Colors::CRLBlueIsh);
                } else {
                    deviceButton = ImGui::Button(buttonIdentifier.c_str(),
                                                 ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2));
                }

                ImGui::PopStyleColor(2);

                auto now = std::chrono::system_clock::now();
                std::chrono::duration<float, std::milli> elapsed_milliseconds = now - lastUpdateTime;

                if (elapsed_milliseconds.count() >= 33.0) {
                    gifFrameIndex++;
                    lastUpdateTime = now;
                }

                if (gifFrameIndex >= m_editor->guiResources().gif.totalFrames)
                    gifFrameIndex = 0;

                ImGui::PopFont();
                ImGui::PopStyleVar(2);

                if (deviceButton) {
                    m_context->multiSense()->setSelectedMultiSenseProfile(profile);
                }

                if (deviceButton &&
                    profile->multiSenseTaskManager->connectionState() != MultiSense::MULTISENSE_CONNECTED) {
                    m_context->multiSense()->connect();
                    m_context->multiSense()->connect();
                } else if (deviceButton) {
                    m_context->multiSense()->disconnect();
                }

                ImGui::EndChild();
            }


            if (removeProfileButtonXClicked) {
                m_context->multiSense()->removeProfile(profileToRemove);
            }


            ImGui::PopStyleColor();
        }

        void drawSideBar() {
            ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));

            ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main);
            // Begin the sidebar as a child window
            ImGui::BeginChild("Sidebar", ImVec2(m_editor->ui()->width, m_editor->ui()->height), false,
                              ImGuiWindowFlags_NoScrollWithMouse);
            drawProfilesInSidebar();
            addDeviceButton();
            // Add version number
            ImGui::SetCursorPos(ImVec2(0.0f, m_editor->ui()->height - 10.0f));
            ImGui::PushFont(m_editor->guiResources().font8);
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
        void onUIRender() override {
            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground;
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(m_editor->ui()->width, m_editor->ui()->height));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("WelcomeScreen", &pOpen, window_flags);
            ImVec2 winSize = ImGui::GetWindowSize();

            // Keep all logic inside each function call. Call each element here for a nice orderly overview
            drawSideBar();
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
    private:
        std::chrono::time_point<std::chrono::system_clock> lastUpdateTime = std::chrono::system_clock::now();

    };

}

#endif //SIDEBAR_H
