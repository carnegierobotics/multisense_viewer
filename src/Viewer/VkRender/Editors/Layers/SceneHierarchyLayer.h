//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_SCENEHIERARCHY_H
#define MULTISENSE_VIEWER_SCENEHIERARCHY_H

#include "Viewer/VkRender/ImGui/Layers/LayerSupport/Layer.h"
#include "Viewer/VkRender/Components/GLTFModelComponent.h"
#include "Viewer/VkRender/Components/CameraGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/VkRender/ImGui/Layers/LayerSupport/LayerUtils.h"

namespace VkRender {


    class SceneHierarchyLayer : public Layer {

    public:
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;


        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        void processEntities(GuiObjectHandles &handles) {
            auto view = handles.m_context->registry().view<TagComponent>(
                    entt::exclude<SkyboxGraphicsPipelineComponent, VkRender::ScriptComponent>);

            // Iterate over entities that have a GLTFModelComponent and a TagComponent
            for (auto entity: view) {
                auto &tag = view.get<TagComponent>(entity);
                // Process each entity here
                processEntity(handles, entity, tag);
            }
        }

        void processEntity(GuiObjectHandles &handles, entt::entity entity, TagComponent &tag) {
            // Your processing logic here
            // This function is called for both component types
            if (ImGui::TreeNodeEx(tag.Tag.c_str(), ImGuiTreeNodeFlags_None)) {
                auto e = Entity(entity, handles.m_context);
                if (e.hasComponent<DefaultGraphicsPipelineComponent2>()) {
                    if (ImGui::SmallButton("Reload Shader")) {
                        e.getComponent<DefaultGraphicsPipelineComponent2>().reloadShaders();
                    }
                }

                if (e.hasComponent<TransformComponent>()) {
                    auto& transform = e.getComponent<TransformComponent>();
                    ImGui::Checkbox("Flip Up", &transform.getFlipUpOption());
                }


                if (ImGui::SmallButton(("Delete ##" + tag.Tag).c_str())) {
                    handles.m_context->markEntityForDestruction(Entity(entity, handles.m_context));
                }

                ImGui::TreePop();
            }
        }

        // Example function to handle file import (you'll need to implement the actual logic)
        void openImportFileDialog(const std::string &fileDescription, const std::string &type) {
            if (!loadFileFuture.valid()) {
                auto &opts = RendererConfig::getInstance().getUserSetting();
                std::string openLoc = Utils::getSystemHomePath().string();
                if (!opts.lastOpenedImportModelFolderPath.empty()) {
                    openLoc = opts.lastOpenedImportModelFolderPath.remove_filename().string();
                }
                loadFileFuture = std::async(VkRender::LayerUtils::selectFile, "Select " + fileDescription + " file",
                                            type, openLoc);
            }
        }

        void rightClickPopup() {
            ImGui::SetNextWindowSize(ImVec2(250.0f, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30.0f,
                                                                    15.0f)); // 20 pixels padding on the left and right, 10 pixels top and bottom

            if (ImGui::BeginPopupContextWindow("right click menu", ImGuiPopupFlags_MouseButtonRight)) {

                // Menu options for loading files
                if (ImGui::MenuItem("Load Wavefront (.obj)")) {
                    openImportFileDialog("Wavefront", "obj");
                }
                if (ImGui::MenuItem("Load glTF 2.0 (.gltf)")) {
                    openImportFileDialog("glTF 2.0", "gltf");
                }
                if (ImGui::MenuItem("Load 3D GS file (.ply)")) {
                    openImportFileDialog("Load 3D GS file", "ply");
                }

                ImGui::EndPopup();
            }
            ImGui::PopStyleVar();  // Reset the padding to previous value

        }

        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {


            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.info->applicationWidth, handles.info->applicationHeight); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoBringToFrontOnFocus;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("SceneHierarchyParent", NULL, window_flags);

            ImGui::Text("Scene hierarchy");
            // Calculate 90% of the available width
            float width = ImGui::GetContentRegionAvail().x * 0.9f;
            // Set a dynamic height based on content, starting with a minimum of 150px
            float height = 150.0f; // Start with your minimum height
            float maxHeight = 600.0f;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main); // Example: Dark grey
            // Create the child window with calculated dimensions and scrolling enabled beyond maxHeight
            ImGui::BeginChild("SceneHierarchyChild", ImVec2(width, (height > maxHeight) ? maxHeight : height), true);


            rightClickPopup();

            processEntities(handles);

            ImGui::EndChild();
            ImGui::PopStyleColor();
            // End the parent window

            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}

#endif //MULTISENSE_VIEWER_SCENEHIERARCHY_H
