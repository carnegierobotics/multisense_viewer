//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_PROPERTIESLAYER
#define MULTISENSE_VIEWER_PROPERTIESLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include <glm/gtc/type_ptr.hpp>  // Include this header for glm::value_ptr

namespace VkRender {


    class PropertiesLayer : public Layer {

    public:
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;


        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 55.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.editorUi->width, handles.editorUi->height); // Size (width, height)
            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoBringToFrontOnFocus;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("PropertiesLayer", NULL, window_flags);

            ImGui::Text("Properties goes here");
            std::shared_ptr<Scene> scene = handles.m_context->activeScene();

            auto view = scene->getRegistry().view<TransformComponent>();
            for (auto& entity : view){
                auto& transform = view.get<TransformComponent>(entity);
                // Display the entity's ID or some other identifier as a headline
                ImGui::Text("Entity ID: %d", static_cast<int>(entity));

                // Get current position and rotation
                glm::vec3 position = transform.getPosition();
                glm::quat rotation = transform.getQuaternion();
                transform.rotation;
                // Input fields for position
                ImGui::DragFloat3(("Position##" + std::to_string(static_cast<double>(entity))).c_str(), glm::value_ptr(position), 0.1f);

                // Input fields for rotation (quaternion)
                ImGui::DragFloat4(("Rotation##" + std::to_string(static_cast<double>(entity))).c_str(), glm::value_ptr(rotation), 0.01f);

                // Apply the new position and rotation back to the transform
                transform.setPosition(position);
                transform.setQuaternion(rotation);

                // Add some space between each entity
                ImGui::Separator();
            }

            ImGui::End();
        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}

#endif //MULTISENSE_VIEWER_PROPERTIESLAYER
