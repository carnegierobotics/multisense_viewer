//
// Created by magnus on 8/23/24.
//


#include "Viewer/Rendering/Editors/Common/SceneHierarchy/SceneHierarchyLayer.h"

#include "Viewer/Rendering/ImGui/IconsFontAwesome6.h"
#include "Viewer/Rendering/Editors/Common/CommonEditorFunctions.h"

namespace VkRender {


    /** Called once upon this object creation**/
    void SceneHierarchyLayer::onAttach() {

    }

/** Called after frame has finished rendered **/
    void SceneHierarchyLayer::onFinishedRender() {

    }


    void SceneHierarchyLayer::drawEntityNode(Entity entity) {
        auto &tag = entity.getComponent<TagComponent>().Tag;

        ImGuiTreeNodeFlags flags = ((m_context->getSelectedEntity() == entity) ? ImGuiTreeNodeFlags_Selected : 0) |
                                   ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_OpenOnArrow |
                                   ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowOverlap;

        if (!entity.hasChildren())
            flags |= ImGuiTreeNodeFlags_Leaf;

        ImGui::PushID((void *) (uint64_t) (uint32_t) entity);



        // Display different color for groups
        bool isGroup = entity.hasComponent<GroupComponent>();
        if (isGroup) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.7f, 0.2f, 1.0f)); // Gold color
        }

        bool opened = ImGui::TreeNodeEx((void *) (uint64_t) (uint32_t) entity, flags, "%s", tag.c_str());

        // Context menu
        bool entityDeleted = false;
        // Begin Popup Context for the current TreeNode (must match the TreeNode ID)
        if (ImGui::BeginPopupContextItem("EntityContextMenu")) {
            if (ImGui::MenuItem("Delete Entity")) {
                // Mark entity as deleted
                entityDeleted = true;
            }
            ImGui::EndPopup();
        }

        // Handle visibility toggle
        if (entity.hasComponent<VisibleComponent>()) {
            ImGui::SameLine();
            auto &visible = entity.getComponent<VisibleComponent>().visible;
            // Use font icons for the button
            ImGui::PushFont(m_editor->guiResources().fontIcons);
            // Set up the button label based on visibility
            std::string label = visible ? ICON_FA_EYE : ICON_FA_EYE_SLASH;
            // Create a unique button ID by appending the entity handle (or other unique value)
            std::string buttonID = "##visibility_" + std::to_string(static_cast<uint32_t>(entity));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0)); // Transparent background
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0, 0, 0, 0)); // Transparent hover background
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0)); // Transparent active background
            if (ImGui::SmallButton((label + buttonID).c_str())) {
                // Toggle visibility when the button is clicked
                visible = !visible;
            }
            ImGui::PopStyleColor(3);
            ImGui::PopFont();
        }

        if (isGroup) {
            ImGui::PopStyleColor();
        }

        // Handle selection
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
            m_context->setSelectedEntity(entity);
        }

        // Drag-and-drop source
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("DND_ENTITY", &entity, sizeof(Entity));
            ImGui::Text("%s", tag.c_str());
            ImGui::EndDragDropSource();
        }

        // Drag-and-drop target
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DND_ENTITY")) {
                Entity droppedEntity = *(Entity *) payload->Data;
                if (entity.hasComponent<GroupComponent>() &&
                    droppedEntity != entity &&
                    !m_context->activeScene()->isDescendantOf(entity, droppedEntity)) {
                    droppedEntity.setParent(entity);
                }
            }
            ImGui::EndDragDropTarget();
        }

        // Recursively draw children
        if (opened) {
            if (entity.hasChildren()) {
                for (auto &child: entity.getChildren()) {
                    drawEntityNode(child);
                }
            }
            ImGui::TreePop();
        }

        if (entityDeleted) {
            m_context->activeScene()->destroyEntityRecursively(entity);
            if (m_context->getSelectedEntity() == entity) {
                m_context->setSelectedEntity(Entity{});
            }
        }

        ImGui::PopID();
    }

    void SceneHierarchyLayer::processEntities() {
        if (m_context->activeScene()) {
            // Get all root entities (entities without a ParentComponent)
            auto view = m_context->activeScene()->getRegistry().view<TagComponent>(entt::exclude<ParentComponent>);
            for (auto entityID: view) {
                Entity entity{entityID, m_context->activeScene().get()};
                drawEntityNode(entity);
            }

            // Right-click on blank space to create entities or groups
            if (ImGui::BeginPopupContextWindow(0, 1)) {
                if (ImGui::MenuItem("Create Empty Entity")) {
                    m_context->activeScene()->createEntity("Empty Entity");
                }
                if (ImGui::MenuItem("Create Group")) {
                    auto groupEntity = m_context->activeScene()->createEntity("New Group");
                    groupEntity.addComponent<GroupComponent>();
                    groupEntity.addComponent<VisibleComponent>(); // For visibility toggling
                }
                ImGui::EndPopup();
            }
        }
    }


/** Handle the file path after selection is complete **/


/** Called once per frame **/
    void SceneHierarchyLayer::onUIRender() {
        // Set window position and size
        ImVec2 window_pos = ImVec2(0.0f, m_editor->ui()->layoutConstants.uiYOffset); // Position (x, y)
        ImVec2 window_size = ImVec2(m_editor->ui()->width, m_editor->ui()->height);  // Size (width, height)

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

        // Calculate the dimensions for the child window
        // Subtract 10.0f to account for 5 px border on each side
        float width = ImGui::GetContentRegionAvail().x - 10.0f;
        float height = ImGui::GetContentRegionAvail().y - 20.0f;

        // Ensure width and height are not negative
        if (width < 0.0f) width = 0.0f;
        if (height < 0.0f) height = 0.0f;

        // Optional: Push a style color for the child window background
        ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLGray424Main); // Example: Dark grey

        // Create the child window with calculated dimensions
        ImGui::BeginChild("SceneHierarchyChild", ImVec2(width, height), true);
        processEntities();

        ImGui::EndChild();

        // Pop the style color if you pushed it
        ImGui::PopStyleColor();

        // End the parent window
        ImGui::End();
    }

/** Called once upon this object destruction **/
    void SceneHierarchyLayer::onDetach() {

    }

}
