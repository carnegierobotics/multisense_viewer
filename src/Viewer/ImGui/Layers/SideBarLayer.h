//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_SIDEBARLAYER_H
#define MULTISENSE_VIEWER_SIDEBARLAYER_H

#include "Viewer/ImGui/Layers/Layer.h"
#include "Viewer/ImGui/Layers/LayerUtils.h"

#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {

    class SideBarLayer : public VkRender::Layer {
    public:

        std::future<LayerUtils::LoadFileInfo> loadFileFuture;
        std::future<std::filesystem::path> folderFuture;

        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        void processEntities(GuiObjectHandles *handles) {
            auto view = handles->m_context->m_registry.view<TagComponent>(
                    entt::exclude<VkRender::SkyboxGraphicsPipelineComponent, VkRender::ScriptComponent>);

            // Iterate over entities that have a GLTFModelComponent and a TagComponent
            for (auto entity: view) {
                auto &tag = view.get<TagComponent>(entity);
                // Process each entity here
                processEntity(handles, entity, tag);
            }
        }

        void processEntity(GuiObjectHandles *handles, entt::entity entity, TagComponent &tag) {
            // Your processing logic here
            // This function is called for both component types
            if (ImGui::TreeNodeEx(tag.Tag.c_str(), ImGuiTreeNodeFlags_None)) {
                auto e = Entity(entity, handles->m_context);
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
                    handles->m_context->markEntityForDestruction(Entity(entity, handles->m_context));
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

        void createSceneHierarchy(GuiObjectHandles *handles) {
            ImGui::Text("Scene hierarchy");
            // Calculate 90% of the available width
            float width = ImGui::GetContentRegionAvail().x * 0.9f;
            // Set a dynamic height based on content, starting with a minimum of 150px
            float height = 150.0f; // Start with your minimum height
            float maxHeight = 600.0f;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main); // Example: Dark grey
            // Create the child window with calculated dimensions and scrolling enabled beyond maxHeight
            ImGui::BeginChild("MyChild", ImVec2(width, (height > maxHeight) ? maxHeight : height), true);


            rightClickPopup();

            processEntities(handles);
            ImGui::EndChild();
            ImGui::PopStyleColor(); // Reset to previous style color
        }

        void otherTab(GuiObjectHandles *handles) {
            // Set a dynamic height based on content, starting with a minimum of 150px
            float height = 50.0f; // Start with your minimum height
            float maxHeight = 500.0f;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main); // Example: Dark grey
            // Create the child window with calculated dimensions and scrolling enabled beyond maxHeight
            ImGui::BeginChild("otherTabChild", ImVec2(-FLT_MIN, (height > maxHeight) ? maxHeight : height), true);

            ImGui::EndChild();
            ImGui::PopStyleColor(); // ImGuiCol_ChildBg
        }


        void cameraTab(GuiObjectHandles *handles) {
            // Set a dynamic height based on content, starting with a minimum of 150px
            float height = 250.0f; // Start with your minimum height
            float maxHeight = 600.0f;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main); // Example: Dark grey
            // Create the child window with calculated dimensions and scrolling enabled beyond maxHeight
            ImGui::BeginChild("cameraTabChild", ImVec2(-FLT_MIN, (height > maxHeight) ? maxHeight : height), true);


            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            const std::string &cameraTag = handles->m_cameraSelection.tag;
            handles->m_cameraSelection.info[cameraTag].reset = false;
            if (ImGui::RadioButton("Arcball", &handles->m_cameraSelection.info[cameraTag].type, 0)) {
                handles->usageMonitor->userClickAction("Arcball", "RadioButton", ImGui::GetCurrentWindow()->Name);
                auto& camera = handles->m_context->getCamera(cameraTag);
                camera.setType(Camera::arcball);
                camera.resetPosition();
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Flycam", &handles->m_cameraSelection.info[cameraTag].type, 1)) {
                handles->usageMonitor->userClickAction("Flycam", "RadioButton", ImGui::GetCurrentWindow()->Name);
                auto& camera = handles->m_context->getCamera(cameraTag);
                camera.setType(Camera::flycam);
                camera.resetPosition();
            }
            ImGui::SameLine();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));
            ImGui::HelpMarker(
                    "Select between arcball or flycam type. Flycam uses Arrow/WASD keys to move camera and mouse + click to rotate");
            ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding

            handles->m_cameraSelection.info[cameraTag].reset |= ImGui::Button(
                    "Reset camera position"); // OR true due to resetCamera may be set by clicking radio buttons above
            if (handles->m_cameraSelection.info[cameraTag].reset) {
                handles->usageMonitor->userClickAction("Reset camera position", "Button",
                                                       ImGui::GetCurrentWindow()->Name);
            }

            std::vector<std::string> cameras;
            for (auto [entity, camera, tag]: handles->m_context->m_registry.view<VkRender::CameraComponent, VkRender::TagComponent>().each()) {
                if (!Utils::isInVector(cameras, tag.Tag))
                    cameras.emplace_back(tag.Tag);
            }

            handles->m_cameraSelection.selected = false;
            if (ImGui::BeginListBox("Cameras", ImVec2(-FLT_MIN, 75.0f))) {
                for (int n = cameras.size() - 1; n >= 0; n--) {
                    const bool is_selected = (handles->m_cameraSelection.currentItemSelected == n);
                    if (ImGui::Selectable(cameras[n].c_str(), is_selected)) {
                        handles->m_cameraSelection.currentItemSelected = n;
                        handles->m_cameraSelection.tag = cameras[n];
                        handles->m_cameraSelection.selected = true;
                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndListBox();
            }

            if (ImGui::Button("Add Camera", ImVec2(150.0f, 25.0f))) {
                std::string tag = "Camera #" + std::to_string(cameras.size());

                auto camera = handles->m_context->createNewCamera(tag, handles->info->width, handles->info->height);
                //e.addComponent<CameraGraphicsPipelineComponent>(&handles->m_context->renderUtils);
                handles->m_cameraSelection.tag = tag;
            }


            if (ImGui::Button("Remove Camera", ImVec2(150.0f, 25.0f))) {
                std::string tag = handles->m_cameraSelection.tag;
                VkRender::Entity entity = handles->m_context->findEntityByName(tag);

                if (entity) {
                    handles->m_context->markEntityForDestruction(entity);
                    // Update the cameras list immediately after deletion
                    cameras.clear();
                    for (auto [ent, camera, tagComponent]: handles->m_context->m_registry.view<VkRender::CameraComponent, VkRender::TagComponent>().each()) {
                        if (!Utils::isInVector(cameras, tagComponent.Tag))
                            cameras.emplace_back(tagComponent.Tag);
                    }
                    // Check if the currently selected camera was deleted
                    if (std::find(cameras.begin(), cameras.end(), tag) == cameras.end()) {
                        // The selected camera was deleted, update the selection
                        if (cameras.empty()) {
                            // No cameras left
                            handles->m_cameraSelection.currentItemSelected = -1;
                            handles->m_cameraSelection.tag.clear();
                        } else {
                            // Select a new camera, preferably the one after the deleted one, or the last if deleted was the last
                            handles->m_cameraSelection.currentItemSelected = std::min(
                                    handles->m_cameraSelection.currentItemSelected, int(cameras.size() - 1));
                            handles->m_cameraSelection.tag = cameras[handles->m_cameraSelection.currentItemSelected];
                        }
                    }
                }
            }


            if (ImGui::Button("Load Cameras", ImVec2(150.0f, 25.0f))) {
                if (!folderFuture.valid()) {
                    auto &opts = RendererConfig::getInstance().getUserSetting();
                    std::string openLoc = Utils::getSystemHomePath().string();
                    if (!opts.lastOpenedFolderPath.empty()) {
                        openLoc = opts.lastOpenedFolderPath.string();
                    }
                    folderFuture = std::async(VkRender::LayerUtils::selectFolder, openLoc);
                }
            }

            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            const std::string &tag = handles->m_cameraSelection.tag;
            if (!tag.empty()) {
                auto& camera = handles->m_context->getCamera(tag);
                    ImGui::Text("Camera: %s", tag.c_str());
                    ImGui::Text("Position: (%.3f, %.3f, %.3f)",
                                static_cast<double>(camera.pose.pos.x),
                                static_cast<double>(camera.pose.pos.y),
                                static_cast<double>(camera.pose.pos.z));

                    ImGui::Text("Q: (%.3f, %.3f, %.3f, %.3f)",
                                static_cast<double>(camera.pose.q.w),
                                static_cast<double>(camera.pose.q.x),
                                static_cast<double>(camera.pose.q.y),
                                static_cast<double>(camera.pose.q.z));
            }
            ImGui::PopStyleColor(); // ImGuiCol_Text


            ImGui::PopStyleColor(); // ImGuiCol_Text
            ImGui::EndChild();
            ImGui::PopStyleColor(); // ImGuiCol_ChildBg

        }

        void createTabBar(GuiObjectHandles *handles) {

            ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
            ImVec2 framePadding = ImGui::GetStyle().FramePadding;
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 5.0f));
            ImGui::PushFont(handles->info->font15);

            if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {

                /// Calculate spaces for centering tab bar text
                int numControlTabs = 2;
                float tabBarWidth = handles->info->sidebarWidth / numControlTabs;
                ImGui::SetNextItemWidth(tabBarWidth);
                std::string tabLabel = "Cameras";

                if (ImGui::BeginTabItem((tabLabel).c_str())) {
                    ImGui::PushFont(handles->info->font13);
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);

                    cameraTab(handles);

                    ImGui::PopStyleVar();//Framepadding
                    ImGui::PopFont();
                    ImGui::EndTabItem();
                }
                if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                    handles->usageMonitor->userClickAction("Preview Control", "tab",
                                                           ImGui::GetCurrentWindow()->Name);
                /// Calculate spaces for centering tab bar text
                ImGui::SetNextItemWidth(tabBarWidth);
                tabLabel = "Other";
                if (ImGui::BeginTabItem((tabLabel).c_str())) {
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);
                    ImGui::PushFont(handles->info->font13);

                    otherTab(handles);

                    ImGui::PopStyleVar();//Framepadding
                    ImGui::PopFont();
                    ImGui::EndTabItem();
                }
                if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                    handles->usageMonitor->userClickAction("Sensor Config", "tab", ImGui::GetCurrentWindow()->Name);

                ImGui::EndTabBar();
            }
            ImGui::PopFont();
            ImGui::PopStyleVar(); // Framepadding
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
            ImGui::SetNextWindowPos(
                    ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->menuBarHeight),
                    ImGuiCond_Always);
            ImGui::SetNextWindowSize(
                    ImVec2(handles->info->sidebarWidth, handles->info->height - handles->info->menuBarHeight));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);

            ImGui::Begin("SideBarLayer", &pOpen, window_flags);

            createSceneHierarchy(handles);
            createTabBar(handles);

            ImGui::Dummy(ImVec2(25.0f, 25.0f));

            VkRender::LayerUtils::createWidgets(handles, WIDGET_PLACEMENT_RENDERER3D);
            handles->startDataCapture = ImGui::Button("Start recording", ImVec2(150.0f, 25.0f));
            handles->stopDataCapture = ImGui::Button("Stop recording", ImVec2(150.0f, 25.0f));

            ImGui::SetCursorPosY(handles->info->height - handles->info->menuBarHeight - 20.0f);

            if (ImGui::Button("Settings",
                              ImVec2(handles->info->sidebarWidth - (ImGui::GetStyle().FramePadding.x * 2), 20.0f))) {
                handles->showDebugWindow = !handles->showDebugWindow;
                handles->usageMonitor->userClickAction("Settings", "button", ImGui::GetCurrentWindow()->Name);
            }

            ImGui::End();
            ImGui::PopStyleColor();
            /// FUTURES
            if (loadFileFuture.valid()) {
                if (loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    auto selectedFile = loadFileFuture.get(); // This will also make the future invalid
                    if (!selectedFile.path.empty()) {
                        // Do something with the selected folder
                        Log::Logger::getInstance()->info("Selected file {}", selectedFile.path.string().c_str());
                        RendererConfig::getInstance().getUserSetting().lastOpenedImportModelFolderPath = selectedFile.path;
                        if (selectedFile.fileType == "ply"){
                            handles->m_paths.update3DGSPath = true;
                        } else if (selectedFile.fileType == "obj"){
                            handles->m_paths.updateObjPath = true;
                        } else if (selectedFile.fileType == "gltf"){
                            handles->m_paths.updateGLTFPath = true;
                        }
                        handles->m_paths.importFilePath = selectedFile.path;
                    }
                }
            } else {
                handles->m_paths.updateObjPath = false;
                handles->m_paths.update3DGSPath = false;
                handles->m_paths.updateGLTFPath = false;
            }

            if (folderFuture.valid()) {
                if (folderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    auto selectedFolder = folderFuture.get(); // This will also make the future invalid
                    if (!selectedFolder.empty()) {
                        // Do something with the selected folder

                        Log::Logger::getInstance()->info("Selected folder {}", selectedFolder.string().c_str());
                        RendererConfig::getInstance().getUserSetting().lastOpenedFolderPath = selectedFolder;
                        handles->m_loadColmapCameras = true;
                        handles->m_paths.loadColMapPosesPath = selectedFolder;
                    }
                }
            }

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_SIDEBARLAYER_H
