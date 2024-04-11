//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_RENDERER3DLEFT_H
#define MULTISENSE_VIEWER_RENDERER3DLEFT_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/ImGui/LayerUtils.h"

#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Entity.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
namespace VkRender {
    class Renderer3DLeft : public Layer {
    public:


        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

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
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(300.0f, handles->info->height));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
            //ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
            ImGui::Begin("Renderer3DLayer", &pOpen, window_flags);

            handles->info->is3DTopBarHovered = ImGui::IsWindowHovered(
                    ImGuiHoveredFlags_RootAndChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup |
                    ImGuiHoveredFlags_AnyWindow);

            if (ImGui::Button("Back", ImVec2(75.0f, 20.0f))) {
                handles->renderer3D = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Settings", ImVec2(125.0f, 20.0f))) {
                handles->showDebugWindow = !handles->showDebugWindow;
                handles->usageMonitor->userClickAction("Settings", "button", ImGui::GetCurrentWindow()->Name);
            }

            ImGui::Dummy(ImVec2(0.0f, 50.0f));

            // Calculate 90% of the available width
            float width = ImGui::GetContentRegionAvail().x * 0.9f;
            // Set a dynamic height based on content, starting with a minimum of 150px
            float height = 150.0f; // Start with your minimum height
            float maxHeight = 500.0f;

            ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424Main); // Example: Dark grey
            // Create the child window with calculated dimensions and scrolling enabled beyond maxHeight
            ImGui::BeginChild("MyChild", ImVec2(width, (height > maxHeight) ? maxHeight : height), true);

            if (ImGui::SmallButton("Add entity")) {

                handles->m_context->createEntity("New Standard Entity");

            }


            if (ImGui::TreeNodeEx("Scene hierarchy", ImGuiTreeNodeFlags_DefaultOpen)) {
                int i = 0;

                for (auto [entity, id, text]: handles->m_context->m_registry.view<VkRender::IDComponent, VkRender::TagComponent>().each()) {
                    if (ImGui::TreeNode(std::to_string(i).c_str(), "%s", text.Tag.c_str())) {
                        ImGui::Text("Some description");
                        ImGui::SameLine();
                        if (ImGui::SmallButton("button")) {
                            Entity entity2(entity, handles->m_context);
                            handles->m_context->destroyEntity(entity2);
                        }
                        ImGui::TreePop();
                    }
                }


                for (const auto &script: handles->renderBlock.scripts) {
                    if (!script.second)
                        continue;
                    // Use SetNextItemOpen() so set the default state of a node to be open. We could
                    // also use TreeNodeEx() with the ImGuiTreeNodeFlags_DefaultOpen flag to achieve the same thing!
                    if (i == 0)
                        ImGui::SetNextItemOpen(true, ImGuiCond_Once);

                    if (ImGui::TreeNode(std::to_string(i).c_str(), "%s", script.first.c_str())) {
                        ImGui::Text("blah blah");
                        ImGui::SameLine();
                        if (ImGui::SmallButton("button")) {}
                        ImGui::TreePop();
                    }
                    ++i;
                }
                ImGui::TreePop();
            }
            ImGui::EndChild();
            ImGui::PopStyleColor(); // Reset to previous style color

            ImGui::End();
            ImGui::PopStyleColor();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
};
DISABLE_WARNING_POP


#endif //MULTISENSE_VIEWER_RENDERER3DLEFT_H
