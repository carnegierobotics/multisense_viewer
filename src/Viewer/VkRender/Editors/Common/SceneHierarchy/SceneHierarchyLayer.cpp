//
// Created by magnus on 8/23/24.
//


#include "Viewer/VkRender/Editors/Common/SceneHierarchy/SceneHierarchyLayer.h"

namespace VkRender {


    /** Called once upon this object creation**/
    void SceneHierarchyLayer::onAttach() {

    }

/** Called after frame has finished rendered **/
    void SceneHierarchyLayer::onFinishedRender() {

    }

    void SceneHierarchyLayer::drawCameraPanel(GuiObjectHandles &handles, Entity &entity) {
        auto &tag = entity.getComponent<TagComponent>();

        auto &camera = entity.getComponent<CameraComponent>();
        std::string label = "Set Active ##" + tag.Tag;
        ImGui::Checkbox(label.c_str(), &handles.shared->setActiveCamera[static_cast<uint32_t>(entity)]);
        if (ImGui::Button("size"))
            camera.camera.setSize(600, 600);
    }


    void SceneHierarchyLayer::drawEntityNode(GuiObjectHandles &handles, Entity entity)
    {
        auto& tag = entity.getComponent<TagComponent>().Tag;
        handles.shared->m_selectedEntity = m_selectionContext;

        ImGuiTreeNodeFlags flags = ((m_selectionContext == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_OpenOnArrow;
        flags |= ImGuiTreeNodeFlags_SpanAvailWidth;
        bool opened = ImGui::TreeNodeEx((void*)(uint64_t)(uint32_t)entity, flags, "%s", tag.c_str());
        if (ImGui::IsItemClicked())
        {
            m_selectionContext = entity;
        }

        bool entityDeleted = false;
        if (ImGui::BeginPopupContextItem())
        {
            if (ImGui::MenuItem("Delete Entity"))
                entityDeleted = true;

            ImGui::EndPopup();
        }

        if (opened)
        {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Leaf;
            bool opened = ImGui::TreeNodeEx((void*)9817239, flags, "%s", tag.c_str());
            if (ImGui::Button("Delete"))
                entityDeleted = true;
            if (opened)
                ImGui::TreePop();
            ImGui::TreePop();
        }

        if (entityDeleted)
        {
            m_scene->destroyEntity(entity);
            if (m_selectionContext == entity) {
                m_selectionContext = {};
            }
        }
        handles.shared->m_selectedEntity = m_selectionContext;

    }

    void SceneHierarchyLayer::processEntities(GuiObjectHandles &handles) {
        if (!m_scene)
            return;

        /*
        auto view = registry.view<TagComponent>();
        ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                           ImGuiTreeNodeFlags_SpanFullWidth;

        if (ImGui::TreeNodeEx(("Scene: " + handles.m_context->activeScene()->getSceneName()).c_str(),
                              treeNodeFlags)) {

            // Iterate over entities that have a GLTFModelComponent and a TagComponent
            bool anyCameraActive = false;
            for (auto entity: view) {
                auto &tag = view.get<TagComponent>(entity);

                if (ImGui::TreeNodeEx(tag.Tag.c_str(),
                                      ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_OpenOnDoubleClick)) {
                    auto e = Entity(entity, handles.m_context->activeScene().get());
                    if (e.hasComponent<DefaultGraphicsPipeline>()) {
                        if (ImGui::SmallButton("Reload Shader")) {
                            //e.getComponent<DefaultGraphicsPipelineComponent>().reloadShaders();
                        }
                    }
                    if (e.hasComponent<TransformComponent>() && !e.hasComponent<CameraComponent>()) {
                        std::string label = "Flip Up #" + tag.Tag;
                        auto &transform = e.getComponent<TransformComponent>();
                        ImGui::Checkbox(label.c_str(), &transform.getFlipUpOption());
                    }
                    if (e.hasComponent<CameraComponent>()) {
                        drawCameraPanel(handles, e);
                    }
                    if (ImGui::SmallButton(("Delete ##" + tag.Tag).c_str())) {
                        handles.m_context->activeScene()->destroyEntity(
                                Entity(entity, handles.m_context->activeScene().get()));
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }

 */
        if (m_scene)
        {
            m_scene->getRegistry().view<entt::entity>().each([&](auto entityID) {
                Entity entity{ entityID, m_scene.get() };
                // Perform your operations with the entity
                drawEntityNode(handles, entity);
            });



            //if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
            //    m_selectionContext = {};

            // Right-click on blank space
            if (ImGui::BeginPopupContextWindow(0, 1))
            {
                if (ImGui::MenuItem("Create Empty Entity"))
                    m_scene->createEntity("Empty Entity");

                ImGui::EndPopup();
            }

        }

    }

    void
    SceneHierarchyLayer::openImportFileDialog(const std::string &fileDescription, const std::vector<std::string> &type,
                                              LayerUtils::FileTypeLoadFlow flow) {
        if (!loadFileFuture.valid()) {
            auto &opts = RendererConfig::getInstance().getUserSetting();
            std::string openLoc = Utils::getSystemHomePath().string();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLoc = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }
            loadFileFuture = std::async(VkRender::LayerUtils::selectFile, "Select " + fileDescription + " file",
                                        type, openLoc, flow);
        }
    }

/** Handle the file path after selection is complete **/
    void
    SceneHierarchyLayer::handleSelectedFile(const LayerUtils::LoadFileInfo &loadFileInfo, GuiObjectHandles &handles) {
        if (!loadFileInfo.path.empty()) {
            if (loadFileInfo.filetype == LayerUtils::OBJ_FILE) {

                // Load into the active scene
                auto entity = handles.m_context->activeScene()->createEntity(loadFileInfo.path.filename().string());
                entity.addComponent<MeshComponent>(loadFileInfo.path);

            } else if (loadFileInfo.filetype == LayerUtils::PLY_3DGS) {
                // Load into the active scene
                auto &registry = handles.m_context->activeScene()->getRegistry();
                auto view = registry.view<GaussianModelComponent>();
                for (auto &entity: view) {
                    handles.m_context->activeScene()->destroyEntity(
                            Entity(entity, handles.m_context->activeScene().get()));
                }
                auto entity = handles.m_context->activeScene()->createEntity(loadFileInfo.path.filename().string());
                entity.addComponent<GaussianModelComponent>(loadFileInfo.path);

            } else if (loadFileInfo.filetype == LayerUtils::PLY_MESH) {
                // Load into the active scene
                auto entity = handles.m_context->activeScene()->createEntity(loadFileInfo.path.filename().string());
                entity.addComponent<MeshComponent>(loadFileInfo.path);

            }


            // Copy the selected file path to wherever it's needed
            auto &opts = RendererConfig::getInstance().getUserSetting();
            opts.lastOpenedImportModelFolderPath = loadFileInfo.path;
            // Additional processing of the file can be done here
            Log::Logger::getInstance()->info("File selected: {}", loadFileInfo.path.filename().string());
        } else {
            Log::Logger::getInstance()->warning("No file selected.");
        }
    }


    void SceneHierarchyLayer::checkFileImportCompletion(GuiObjectHandles &handles) {
        if (loadFileFuture.valid() &&
            loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            LayerUtils::LoadFileInfo loadFileInfo = loadFileFuture.get(); // Get the result from the future
            handleSelectedFile(loadFileInfo, handles);
        }
    }

    void SceneHierarchyLayer::rightClickPopup() {
        ImGui::SetNextWindowSize(ImVec2(250.0f, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30.0f,
                                                                15.0f)); // 20 pixels padding on the left and right, 10 pixels top and bottom

        if (ImGui::BeginPopupContextWindow("right click menu", ImGuiPopupFlags_MouseButtonRight)) {

            // Menu options for loading files
            if (ImGui::MenuItem("Load Wavefront (.obj)")) {
                std::vector<std::string> types{".obj"};

                openImportFileDialog("Wavefront", types, LayerUtils::OBJ_FILE);
            }

            if (ImGui::MenuItem("Load Mesh ply file")) {
                std::vector<std::string> types{".ply"};
                openImportFileDialog("Load mesh file", types, LayerUtils::PLY_MESH);
            }
            if (ImGui::MenuItem("Load 3D GS file (.ply)")) {
                std::vector<std::string> types{".ply"};
                openImportFileDialog("Load 3D GS file", types, LayerUtils::PLY_3DGS);
            }

            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();  // Reset the padding to previous value

    }


/** Called once per frame **/
    void SceneHierarchyLayer::onUIRender(VkRender::GuiObjectHandles &handles) {


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
    void SceneHierarchyLayer::onDetach() {

    }

}