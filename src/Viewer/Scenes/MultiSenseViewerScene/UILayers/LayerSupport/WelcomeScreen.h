#ifndef WELCOMESCREEN_H
#define WELCOMESCREEN_H
#include <Viewer/ImGui/Layers/LayerSupport/Layer.h>

namespace VkRender {
    static void drawWelcomeScreen(VkRender::GuiObjectHandles &uiContext) {
        // IF we do not have any active cameras we can just draw welcome screen

        // Welcome screen
        ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLCoolGray);
        ImGui::SetCursorPos(ImVec2(uiContext.info->sidebarWidth, 0.0f));
        ImGui::BeginChild("WelcomeScreen",
                          ImVec2(uiContext.info->width - uiContext.info->sidebarWidth, uiContext.info->height),
                          false,
                          ImGuiWindowFlags_NoScrollWithMouse);


        ImGui::EndChild();
        ImGui::PopStyleColor();
    }
}

#endif //WELCOMESCREEN_H
