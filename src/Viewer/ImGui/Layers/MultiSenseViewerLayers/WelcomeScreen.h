#ifndef WELCOMESCREEN_H
#define WELCOMESCREEN_H
#include <Viewer/ImGui/Layers/LayerSupport/Layer.h>

static void drawWelcomeScreen(VkRender::GuiObjectHandles *uiContext) {

    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray421);
    // Begin the sidebar as a child window
    ImGui::SetNextWindowPos(ImVec2(uiContext->info->sidebarWidth, 0.0f));
    ImGui::BeginChild("WelcomeScreen", ImVec2(uiContext->info->width - uiContext->info->sidebarWidth, uiContext->info->height), false,
                      ImGuiWindowFlags_NoScrollWithMouse);

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

#endif //WELCOMESCREEN_H
