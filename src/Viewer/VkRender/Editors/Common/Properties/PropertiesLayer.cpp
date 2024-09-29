//
// Created by magnus on 8/23/24.
//

#include "Viewer/VkRender/Editors/Common/Properties/PropertiesLayer.h"
#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Core/Entity.h"

namespace VkRender {


    /** Called once upon this object creation**/
    void PropertiesLayer::onAttach() {

    }

/** Called after frame has finished rendered **/
    void PropertiesLayer::onFinishedRender() {

    }

    void PropertiesLayer::setScene(std::shared_ptr<Scene> scene) {
        Layer::setScene(scene);

        // Reset the selection context
        m_selectionContext = Entity(); // reset selectioncontext
    }

    void PropertiesLayer::drawVec3Control(const std::string &label, glm::vec3 &values, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        ImGuiIO &io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

        float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize))
            values.x = resetValue;
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::DragFloat("##X", &values.x, 0.1f * speed, 0.0f, 0.0f, "%.2f");
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Y", buttonSize))
            values.y = resetValue;
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::DragFloat("##Y", &values.y, 0.1f * speed, 0.0f, 0.0f, "%.2f");
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Z", buttonSize))
            values.z = resetValue;
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::DragFloat("##Z", &values.z, 0.1f * speed, 0.0f, 0.0f, "%.2f");
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();
    }
    void PropertiesLayer::drawFloatControl(const std::string &label, float &value, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        ImGuiIO &io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

        float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize))
            value = resetValue;
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        ImGui::DragFloat("##X", &value, 0.1f * speed, 0.0f, 0.0f, "%.2f");
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();
    }

    template<typename T, typename UIFunction>
    void PropertiesLayer::drawComponent(const std::string &name, Entity entity, UIFunction uiFunction) {
        const ImGuiTreeNodeFlags treeNodeFlags =
                ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth |
                ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap;
        if (entity.hasComponent<T>()) {
            auto &component = entity.getComponent<T>();
            ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
            float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
            ImGui::Separator();
            bool open = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), treeNodeFlags, "%s", name.c_str());
            ImGui::PopStyleVar();
            ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);

            if (ImGui::Button("+", ImVec2{lineHeight, lineHeight})) {
                ImGui::OpenPopup("ComponentSettings");
            }

            bool removeComponent = false;
            if (ImGui::BeginPopup("ComponentSettings")) {
                if (ImGui::MenuItem("Remove component"))
                    removeComponent = true;

                ImGui::EndPopup();
            }

            if (open) {
                uiFunction(component);
                ImGui::TreePop();
            }

            if (removeComponent)
                entity.removeComponent<T>();
        }
    }

    void PropertiesLayer::drawComponents(Entity entity) {

        if (ImGui::Button("Add Component"))
            ImGui::OpenPopup("AddComponent");

        if (ImGui::BeginPopup("AddComponent")) {
            displayAddComponentEntry<CameraComponent>("Camera");
            displayAddComponentEntry<ScriptComponent>("Script");
            displayAddComponentEntry<TextComponent>("Text Component");
            displayAddComponentEntry<TransformComponent>("Transform Component");

            displayAddComponentEntry<MeshComponent>("MeshComponent");

            ImGui::EndPopup();
        }

        drawComponent<TransformComponent>("Transform", entity, [](auto &component) {
            drawVec3Control("Translation", component.getPosition());
            drawVec3Control("Rotation",  component.getRotation(), 0.0f, 2.0f);
            drawVec3Control("Scale", component.getScale(), 1.0f);
        });
        drawComponent<CameraComponent>("Camera", entity, [](auto &component) {

            drawFloatControl("Field of View", component().fov(), 1.0f);
            component().updateProjectionMatrix();

        });

        drawComponent<MeshComponent>("Mesh", entity, [this](auto &component) {
            ImGui::Text("MeshFile:");
            ImGui::Text("%s", component.getModelPath().string().c_str());
            // Load mesh from file here:
            if (ImGui::Button("Load .obj file")) {
                std::vector<std::string> types{".obj"};
                openImportFileDialog("Wavefront", types, LayerUtils::OBJ_FILE);
            }


        });

    }

    void PropertiesLayer::setSelectedEntity(Entity entity) {
        m_selectionContext = entity;
    }

/** Called once per frame **/
    void PropertiesLayer::onUIRender() {

        setSelectedEntity(m_editor->ui()->shared->m_selectedEntity);

        ImVec2 window_pos = ImVec2(0.0f, m_editor->ui()->layoutConstants.uiYOffset); // Position (x, y)
        ImVec2 window_size = ImVec2(m_editor->ui()->width, m_editor->ui()->height); // Size (width, height)
// Set window flags to remove decorations
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoBringToFrontOnFocus;

// Set next window position and size
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

// Create the parent window
        ImGui::Begin("PropertiesLayer", NULL, window_flags);

        ImGui::Text("Entity Properties");
        std::shared_ptr<Scene> scene = m_context->activeScene();


        if (m_selectionContext) {
            drawComponents(m_selectionContext);
        }

        checkFileImportCompletion();

        /*
        auto view = scene->getRegistry().view<TransformComponent, TagComponent>();
        for (auto &entity: view) {
            auto &transform = view.get<TransformComponent>(entity);
// Display the entity's ID or some other identifier as a headline
            ImGui::Text("Entity: %s", view.get<TagComponent>(entity).Tag.c_str());

// Get current position and rotation
            glm::vec3 &position = transform.getPosition();
            glm::vec3 euler = transform.getEuler();
// Input fields for position
            ImGui::DragFloat3(("Position##" + std::to_string(static_cast<double>(entity))).c_str(),
                              glm::value_ptr(position), 0.1f);

// Input fields for rotation (quaternion)
            ImGui::DragFloat3(("Rotation##" + std::to_string(static_cast<double>(entity))).c_str(),
                              glm::value_ptr(euler), 1.0f);

            transform.setEuler(euler.x, euler.y, euler.z);
// Add some space between each entity
            ImGui::Separator();
            */

        ImGui::End();
    }

/** Called once upon this object destruction **/
    void PropertiesLayer::onDetach() {

    }

    template<typename T>
    void PropertiesLayer::displayAddComponentEntry(const std::string &entryName) {
        if (!m_selectionContext.hasComponent<T>()) {
            if (ImGui::MenuItem(entryName.c_str())) {
                m_selectionContext.addComponent<T>();
                ImGui::CloseCurrentPopup();
            }
        }
    }

    void
    PropertiesLayer::handleSelectedFile(const LayerUtils::LoadFileInfo &loadFileInfo) {
        if (!loadFileInfo.path.empty()) {
            if (loadFileInfo.filetype == LayerUtils::OBJ_FILE) {
                // Load into the active scene
                auto &meshComponent = m_selectionContext.getComponent<MeshComponent>();
                meshComponent.loadOBJ(loadFileInfo.path);

            } else if (loadFileInfo.filetype == LayerUtils::PLY_3DGS) {
                // Load into the active scene
                auto &registry = m_context->activeScene()->getRegistry();
                auto view = registry.view<GaussianModelComponent>();
                for (auto &entity: view) {
                    m_context->activeScene()->destroyEntity(
                            Entity(entity, m_context->activeScene().get()));
                }
                auto entity = m_context->activeScene()->createEntity(loadFileInfo.path.filename().string());
                entity.addComponent<GaussianModelComponent>(loadFileInfo.path);
            } else if (loadFileInfo.filetype == LayerUtils::PLY_MESH) {
                // Load into the active scene
                auto entity = m_context->activeScene()->createEntity(loadFileInfo.path.filename().string());
                entity.addComponent<MeshComponent>(loadFileInfo.path);
            }
            // Copy the selected file path to wherever it's needed
            auto &opts = ApplicationConfig::getInstance().getUserSetting();
            opts.lastOpenedImportModelFolderPath = loadFileInfo.path;
            // Additional processing of the file can be done here
            Log::Logger::getInstance()->info("File selected: {}", loadFileInfo.path.filename().string());
        } else {
            Log::Logger::getInstance()->warning("No file selected.");
        }
    }

    void PropertiesLayer::checkFileImportCompletion() {
        if (loadFileFuture.valid() &&
            loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            LayerUtils::LoadFileInfo loadFileInfo = loadFileFuture.get(); // Get the result from the future
            handleSelectedFile(loadFileInfo);
        }
    }

    void PropertiesLayer::openImportFileDialog(const std::string &fileDescription,
                                               const std::vector<std::string> &type,
                                               LayerUtils::FileTypeLoadFlow flow) {
        if (!loadFileFuture.valid()) {
            auto &opts = ApplicationConfig::getInstance().getUserSetting();
            std::string openLoc = Utils::getSystemHomePath().string();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLoc = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }
            loadFileFuture = std::async(VkRender::LayerUtils::selectFile, "Select " + fileDescription + " file",
                                        type, openLoc, flow);
        }
    }

}