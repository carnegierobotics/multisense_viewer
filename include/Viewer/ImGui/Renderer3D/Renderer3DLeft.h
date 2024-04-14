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
#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"

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

        void processEntities(GuiObjectHandles *handles) {
            // Create a view for entities with a GLTFModelComponent and a TagComponent
            auto gltfView = handles->m_context->m_registry.view<VkRender::GLTFModelComponent, TagComponent>(entt::exclude<RenderResource::SkyboxGraphicsPipelineComponent>);

            // Iterate over entities that have a GLTFModelComponent and a TagComponent
            for (auto entity : gltfView) {
                auto& model = gltfView.get<VkRender::GLTFModelComponent>(entity);
                auto& tag = gltfView.get<TagComponent>(entity);

                // Process each entity here
                processEntity(handles, entity, model, tag);
            }

            // Create a view for entities with a CustomModelComponent and a TagComponent
            auto customView = handles->m_context->m_registry.view<VkRender::CustomModelComponent, TagComponent>();

            // Iterate over entities that have a CustomModelComponent and a TagComponent
            for (auto entity : customView) {
                // To avoid processing entities with both components twice, check if they were already processed
                if (!gltfView.contains(entity)) {
                    auto& model = customView.get<VkRender::CustomModelComponent>(entity);
                    auto& tag = customView.get<TagComponent>(entity);

                    // Process each entity here
                    processEntity(handles, entity, model, tag);
                }
            }
        }

        void processEntity(GuiObjectHandles *handles, entt::entity entity, auto& model, TagComponent& tag) {
            // Your processing logic here
            // This function is called for both component types
            if (ImGui::TreeNodeEx(tag.Tag.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Features coming..");
                ImGui::SameLine();
                if (ImGui::SmallButton(("Delete ##" + tag.Tag).c_str())) {
                    handles->m_context->markEntityForDestruction(Entity(entity, handles->m_context));
                }
                ImGui::TreePop();
            }
        }

        /** Called once per frame **/
        void onUIRender(GuiObjectHandles *handles) override {
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

            processEntities(handles);
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
