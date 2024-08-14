//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_SCENEHIERARCHY_H
#define MULTISENSE_VIEWER_SCENEHIERARCHY_H

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/ImGui/LayerUtils.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"

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
            auto scene = handles.m_context->activeScene();
            if (!scene)
                return;
            auto &registry = scene->getRegistry();

            auto view = registry.view<TagComponent>();
            ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnDoubleClick;

            if (ImGui::TreeNodeEx(("Scene: " + handles.m_context->activeScene()->getSceneName()).c_str(), treeNodeFlags)) {

                // Iterate over entities that have a GLTFModelComponent and a TagComponent
                for (auto entity: view) {
                    auto &tag = view.get<TagComponent>(entity);
                    processEntity(handles, entity, tag);
                }
                ImGui::TreePop();
            }
        }

        void processEntity(GuiObjectHandles &handles, entt::entity entity, TagComponent &tag) {
            // Your processing logic here
            // This function is called for both component types
            if (ImGui::TreeNodeEx(tag.Tag.c_str(), ImGuiTreeNodeFlags_None)) {
                auto e = Entity(entity, handles.m_context->activeScene().get());
                if (e.hasComponent<DefaultGraphicsPipelineComponent>()) {
                    if (ImGui::SmallButton("Reload Shader")) {
                        //e.getComponent<DefaultGraphicsPipelineComponent>().reloadShaders();
                    }
                }
                if (e.hasComponent<TransformComponent>()) {
                    auto &transform = e.getComponent<TransformComponent>();
                    ImGui::Checkbox("Flip Up", &transform.getFlipUpOption());
                }
                if (ImGui::SmallButton(("Delete ##" + tag.Tag).c_str())) {
                    handles.m_context->activeScene()->destroyEntity(Entity(entity, handles.m_context->activeScene().get()));
                }
                ImGui::TreePop();
            }

            // Check if the other tree is hovered

        }

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

        /** Handle the file path after selection is complete **/
        void handleSelectedFile(const LayerUtils::LoadFileInfo &loadFileInfo, GuiObjectHandles& handles) {
            if (!loadFileInfo.path.empty()) {
                // Copy the selected file path to wherever it's needed
                auto &opts = RendererConfig::getInstance().getUserSetting();
                opts.lastOpenedImportModelFolderPath = loadFileInfo.path;
                // Load into the active scene
                auto entity = handles.m_context->activeScene()->createEntity(loadFileInfo.path.filename().string());
                entity.addComponent<OBJModelComponent>(loadFileInfo.path);
                // Additional processing of the file can be done here
                Log::Logger::getInstance()->info("File selected: {}",  loadFileInfo.path.filename().string());
            } else {
                Log::Logger::getInstance()->warning("No file selected.");
            }
        }


        void checkFileImportCompletion(GuiObjectHandles& handles) {
            if (loadFileFuture.valid() && loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                LayerUtils::LoadFileInfo selectedFilePath = loadFileFuture.get(); // Get the result from the future
                handleSelectedFile(selectedFilePath, handles);
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
                /*
                if (ImGui::MenuItem("Load glTF 2.0 (.gltf)")) {
                    openImportFileDialog("glTF 2.0", "gltf");
                }
                if (ImGui::MenuItem("Load 3D GS file (.ply)")) {
                    openImportFileDialog("Load 3D GS file", "ply");
                }
                */
                ImGui::EndPopup();
            }
            ImGui::PopStyleVar();  // Reset the padding to previous value

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {


            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, handles.info->menuBarHeight); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.editorUi->width, handles.editorUi->height); // Size (width, height)

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
            ImGui::SetCursorPosX((window_size.x - width) / 2);
            ImGui::BeginChild("SceneHierarchyChild", ImVec2(width, (height > maxHeight) ? maxHeight : height), true);

            rightClickPopup();

            processEntities(handles);

            ImGui::EndChild();
            ImGui::PopStyleColor();
            // End the parent window

            ImGui::End();

            checkFileImportCompletion(handles);
        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}

#endif //MULTISENSE_VIEWER_SCENEHIERARCHY_H
