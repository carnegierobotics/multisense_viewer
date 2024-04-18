//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_RENDERER3DBOTTOM_H
#define MULTISENSE_VIEWER_RENDERER3DBOTTOM_H

#include "Viewer/ImGui/Layer.h"
/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
namespace VkRender {


    class Renderer3DBottom : public VkRender::Layer {
    public:


        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        void setCusorToColumn(uint32_t column, float yOffset = 0.0f) {

            float columnSpacing = 200.0f;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + columnSpacing * column);
            ImGui::SetCursorPosY(ImGui::GetStyle().WindowPadding.y + yOffset);
        }

        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles *handles) override {
            if (!handles->renderer3D)
                return;
            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse |
                    ImGuiWindowFlags_NoResize;
            ImGui::SetNextWindowPos(ImVec2(300.0f, handles->info->height - 150.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(handles->info->width - (300.0f * 2), 150.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);

            ImGui::Begin("Renderer3DBottom", &pOpen, window_flags);

            VkRender::LayerUtils::WidgetPosition pos;
            pos.paddingX = 5.0f;
            pos.maxElementWidth = 230.0f;
            VkRender::LayerUtils::createWidgets(handles, WIDGET_PLACEMENT_RENDERER3D);

            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            handles->camera.reset = false;
            if (ImGui::RadioButton("Arcball", &handles->camera.type, 0)) {
                handles->usageMonitor->userClickAction("Arcball", "RadioButton", ImGui::GetCurrentWindow()->Name);
                handles->camera.reset = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Flycam", &handles->camera.type, 1)) {
                handles->usageMonitor->userClickAction("Flycam", "RadioButton", ImGui::GetCurrentWindow()->Name);
                handles->camera.reset = true;
            }
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));
            ImGui::HelpMarker(
                    "Select between arcball or flycam type. Flycam uses Arrow/WASD keys to move camera and mouse + click to rotate");
            ImGui::PopStyleVar();
            handles->camera.reset |= ImGui::Button(
                    "Reset camera position"); // OR true due to resetCamera may be set by clicking radio buttons above
            if (handles->camera.reset) {
                handles->usageMonitor->userClickAction("Reset camera position", "Button",
                                                       ImGui::GetCurrentWindow()->Name);
            }

            setCusorToColumn(1);
            auto cameraView = handles->m_context->m_registry.view<VkRender::CameraComponent>();

            std::vector<std::string> cameras;
            for (auto entity: cameraView) {
                auto &camera = cameraView.get<VkRender::CameraComponent>(entity);
                if (!Utils::isInVector(cameras, camera.tag))
                    cameras.emplace_back(camera.tag);
            }
            ImGui::SetNextItemWidth(100.0f);
            static int item_current_idx = 0; // Here we store our selection data as an index.
            if (ImGui::BeginListBox("Cameras")) {
                for (int n = 0; n < cameras.size(); n++) {
                    const bool is_selected = (item_current_idx == n);
                    if (ImGui::Selectable(cameras[n].c_str(), is_selected))
                        item_current_idx = n;

                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }
            setCusorToColumn(2);
            if (ImGui::Button("Add Camera", ImVec2(150.0f, 25.0f))) {
                auto e = handles->m_context->createEntity("New Entity");
                e.addComponent<CameraComponent>("Camera #" + std::to_string(cameras.size() + 1));
            }
            setCusorToColumn(2, ImGui::GetCursorPosY());

            if (ImGui::Button("Remove Camera", ImVec2(150.0f, 25.0f))) {
                handles->m_context->destroyEntity(handles->m_context->findEntityByName("New Entity"));
            }

            ImGui::PopStyleColor(); // text white


            ImGui::End();
            ImGui::PopStyleColor();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
};
#endif //MULTISENSE_VIEWER_RENDERER3DBOTTOM_H
