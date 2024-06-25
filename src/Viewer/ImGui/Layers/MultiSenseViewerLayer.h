//
// Created by magnus on 6/24/24.
//

#ifndef WELCOMESCREENLAYER_H
#define WELCOMESCREENLAYER_H

#include "MultiSenseViewerLayers/Sidebar.h"
#include "MultiSenseViewerLayers/WelcomeScreen.h"
#include "Viewer/ImGui/Layers/LayerSupport/Layer.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {

    class MultiSenseViewerLayer : public Layer {
    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }




        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles *uiContext) override {
            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground;
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(uiContext->info->width - uiContext->info->sidebarWidth, uiContext->info->height));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("WelcomeScreen", &pOpen, window_flags);
            ImVec2 winSize = ImGui::GetWindowSize();

            // Keep all logic inside each function call. Call each element here for a nice orderly overview
            drawSideBar(uiContext);
            // Draw welcome screen
            drawWelcomeScreen(uiContext);
            // Draw Main preview page, after we obtained a successfull connection
            // Draw the 2D view (Camera stream, layouts etc..)
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
#endif //WELCOMESCREENLAYER_H
