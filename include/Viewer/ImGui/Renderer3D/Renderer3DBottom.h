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
            const std::string &cameraTag = handles->m_cameraSelection.tag;
            handles->m_cameraSelection.info[cameraTag].reset = false;
            if (ImGui::RadioButton("Arcball", &handles->m_cameraSelection.info[cameraTag].type, 0)) {
                handles->usageMonitor->userClickAction("Arcball", "RadioButton", ImGui::GetCurrentWindow()->Name);
                handles->m_cameraSelection.info[cameraTag].reset = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Flycam", &handles->m_cameraSelection.info[cameraTag].type, 1)) {
                handles->usageMonitor->userClickAction("Flycam", "RadioButton", ImGui::GetCurrentWindow()->Name);
                handles->m_cameraSelection.info[cameraTag].reset = true;
            }
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));
            ImGui::HelpMarker(
                    "Select between arcball or flycam type. Flycam uses Arrow/WASD keys to move camera and mouse + click to rotate");
            ImGui::PopStyleVar();
            handles->m_cameraSelection.info[cameraTag].reset |= ImGui::Button(
                    "Reset camera position"); // OR true due to resetCamera may be set by clicking radio buttons above
            if (handles->m_cameraSelection.info[cameraTag].reset) {
                handles->usageMonitor->userClickAction("Reset camera position", "Button",
                                                       ImGui::GetCurrentWindow()->Name);
            }

            setCusorToColumn(1);
            // use a range-for
            std::vector<std::string> cameras;
            for (auto [entity, camera, tag]: handles->m_context->m_registry.view<VkRender::CameraComponent, VkRender::TagComponent>().each()) {
                if (!Utils::isInVector(cameras, tag.Tag))
                    cameras.emplace_back(tag.Tag);

            }

            ImGui::SetNextItemWidth(100.0f);
            static int item_current_idx = 0; // Here we store our selection data as an index.
            if (ImGui::BeginListBox("Cameras")) {
                for (int n = 0; n < cameras.size(); n++) {
                    const bool is_selected = (item_current_idx == n);
                    if (ImGui::Selectable(cameras[n].c_str(), is_selected)) {
                        item_current_idx = n;
                        handles->m_cameraSelection.tag = cameras[n];
                    }

                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }
            setCusorToColumn(2);
            if (ImGui::Button("Add Camera", ImVec2(150.0f, 25.0f))) {
                std::string tag = "Camera #" + std::to_string(cameras.size());

                auto e = handles->m_context->createEntity(tag);
                e.addComponent<CameraComponent>();
                handles->m_context->cameras[tag] = Camera(handles->info->width, handles->info->height);
                handles->m_cameraSelection.tag = tag;
            }
            setCusorToColumn(2, ImGui::GetCursorPosY());

            if (ImGui::Button("Remove Camera", ImVec2(150.0f, 25.0f))) {
                std::string tag = handles->m_cameraSelection.tag;
                VkRender::Entity entity = handles->m_context->findEntityByName(tag);

                if (entity) {
                    handles->m_context->destroyEntity(entity);
                    // Update the cameras list immediately after deletion
                    cameras.clear();
                    for (auto [entity, camera, tagComponent]: handles->m_context->m_registry.view<VkRender::CameraComponent, VkRender::TagComponent>().each()) {
                        if (!Utils::isInVector(cameras, tagComponent.Tag))
                            cameras.emplace_back(tagComponent.Tag);
                    }
                    // Check if the currently selected camera was deleted
                    if (std::find(cameras.begin(), cameras.end(), tag) == cameras.end()) {
                        // The selected camera was deleted, update the selection
                        if (cameras.size() == 0) {
                            // No cameras left
                            item_current_idx = -1;
                            handles->m_cameraSelection.tag.clear();
                        } else {
                            // Select a new camera, preferably the one after the deleted one, or the last if deleted was the last
                            item_current_idx = std::min(item_current_idx, int(cameras.size() - 1));
                            handles->m_cameraSelection.tag = cameras[item_current_idx];
                        }
                    }
                }
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
