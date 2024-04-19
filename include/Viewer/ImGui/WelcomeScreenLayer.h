//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_WELCOMESCREENLAYER_H
#define MULTISENSE_VIEWER_WELCOMESCREENLAYER_H


#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"
// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

class WelcomeScreenLayer : public VkRender::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        bool shouldDraw = true; //
        for (const auto& dev: handles->devices){
            if (dev.state == VkRender::CRL_STATE_ACTIVE)
                shouldDraw = false;
        }
        if (!shouldDraw || handles->renderer3D)
            return;

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("WelcomeScreen", &pOpen, window_flags);
        ImVec2 winSize = ImGui::GetWindowSize();

        ImVec2 btnSize(120.0f, 50.0f);

        // Btn X length is: 150 * 3 + 20 = 470
        // So I should start drawing at winSize / 2 - 235.0f
        ImGui::SetCursorPos(ImVec2(winSize.x / 2.0f - 235.0f, winSize.y / 2.0f - 25.0f));
        ImGui::PushFont(handles->info->font15);
        if(ImGui::Button("Add Device", btnSize)){
            handles->openAddDevicePopup = true;
            handles->usageMonitor->userClickAction("ADD_DEVICE", "button", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::SameLine(0.0f, 20.0f);
        if(ImGui::Button("3D Renderer", btnSize)){
            // Open 3D Renderer with a basic scene.
            // Get rid of sidebar
            //
            handles->renderer3D = true;
        }
        ImGui::SameLine(0.0f, 20.0f);

        if(ImGui::Button("MS Renderer", btnSize)){
                handles->usageMonitor->userClickAction("MultiSense RENDERER", "Button", ImGui::GetCurrentWindow()->Name);
                // Add test device to renderer if not present
                bool exists = false;
                for (const auto &device: handles->devices) {
                    if (device.cameraName == "Simulated device")
                        exists = true;
                }
                if (!exists) {
                    VkRender::Device testDevice;
                    testDevice.name = "AccuRender Profile";
                    Utils::initializeUIDataBlockWithTestData(testDevice);
                    handles->devices.emplace_back(testDevice);
                    Log::Logger::getInstance()->info("Adding a test device to the profile section");
                }
        }

        ImGui::PopFont();
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();

    }


    /** Called once upon this object destruction **/
    void onDetach() override {

    }
};

DISABLE_WARNING_POP


#endif //MULTISENSE_VIEWER_WELCOMESCREENLAYER_H
