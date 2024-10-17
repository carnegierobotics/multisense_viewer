//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_MENULAYER_H
#define MULTISENSE_VIEWER_MENULAYER_H

#include <Viewer/Scenes/SceneSerializer.h>

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Editors/Common/CommonEditorFunctions.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class MenuLayer : public VkRender::Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender() override {


            // Push style variables
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 5.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 10.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 3.0f));

            // Set colors
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

            float menuBarHeight = 25.0f;
            ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, menuBarHeight));
            ImGui::BeginMainMenuBar();
            if (ImGui::BeginMenu("File")) {
                // Projects Menu
                if (ImGui::BeginMenu("Projects")) {
                    bool isDefaultProject = m_context->isCurrentProject("MultiSense Editor");
                    bool isMultiSenseProject = m_context->isCurrentProject("MultiSense Viewer");

                    if (ImGui::MenuItem("MultiSense Viewer", nullptr, isMultiSenseProject)) {
                        if (!isMultiSenseProject) {
                            m_context->loadProject(Utils::getProjectFileFromName("MultiSense Viewer"));
                            ImGui::SetCurrentContext(m_context->getMainUIContext());
                        }
                    }

                    if (ImGui::MenuItem("MultiSense Editor", nullptr, isDefaultProject)) {
                        if (!isDefaultProject) {
                            m_context->loadProject(Utils::getProjectFileFromName("MultiSense Editor"));
                            ImGui::SetCurrentContext(m_context->getMainUIContext());
                        }
                    }
                    ImGui::EndMenu();  // End the Projects submenu
                }

                // Scenes Menu
                if (ImGui::BeginMenu("Scenes")) {
                    bool isDefaultScene = m_context->isCurrentScene("Default Scene");
                    bool isMultiSenseScene = m_context->isCurrentScene("MultiSense Viewer Scene");

                    if (ImGui::MenuItem("Save Scene", nullptr, isDefaultScene)) {

                        //SceneSerializer serializer(m_context->activeScene());
                        //serializer.serialize(Utils::getAssetsPath() / "Scenes" / "Example.multisense");


                    }
                    if (ImGui::MenuItem("Save Scene As..", nullptr, isDefaultScene)) {

                        //SceneSerializer serializer(m_context->activeScene());
                        //serializer.serialize(Utils::getAssetsPath() / "Scenes" / "Example.multisense");


                    }
                    if (ImGui::MenuItem("Load Scene file", nullptr, isMultiSenseScene)) {

                        std::vector<std::string> types{".multisense"};
                        EditorUtils::openImportFileDialog("Load Scene", types, LayerUtils::SCENE_FILE, &loadFileFuture);

                    }
                    ImGui::EndMenu();  // End the Scenes submenu
                }

                if (ImGui::MenuItem("Quit")) {
                    // Handle quitting the application
                    m_context->closeApplication();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Console", nullptr, &m_editor->ui()->showDebugWindow);
                ImGui::EndMenu();
            }


            ImGui::EndMainMenuBar();

            checkFileImportCompletion();

            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor(5);


        }

        void
        handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo &loadFileInfo) {
            if (!loadFileInfo.path.empty()) {
                switch (loadFileInfo.filetype) {
                    case LayerUtils::SCENE_FILE: {
                        SceneSerializer serializer(m_context->activeScene());
                        serializer.deserialize(loadFileInfo.path);
                    }
                        break;
                    default:
                        break;

                }
            }
        }

        void checkFileImportCompletion() {
            if (loadFileFuture.valid() &&
                loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                LayerUtils::LoadFileInfo loadFileInfo = loadFileFuture.get(); // Get the result from the future
                handleSelectedFileOrFolder(loadFileInfo);
            }
        }

        /** Called once upon this object destruction **/
        void onDetach()
        override {

        }

    private:
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;

    };

}
#endif //MULTISENSE_VIEWER_MENULAYER_H
