//
// Created by magnus on 8/23/24.
//

#include <glm/gtc/type_ptr.hpp>  // for glm::value_ptr

#include "Viewer/Rendering/Editors/Properties/PropertiesLayer.h"

#include <Viewer/Rendering/Components/ScriptableComponent.h>
#include <Viewer/Scenes/CameraController.h>
#include <Viewer/Scripts/VectorScripts.h>
#include <Viewer/Scripts/Rays/ContributionRay.h>
#include <Viewer/Scripts/Rays/Emitter.h>
#include <Viewer/Scripts/Rays/GradientRay.h>
#include <Viewer/Scripts/Rays/IntensityGradientRay.h>
#include <Viewer/Scripts/Rays/SurfaceNormal.h>

#include "Viewer/Rendering/Components/LightSourceComponent.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/QuadricCollectionComponent.h"
#include "Viewer/Rendering/Components/CameraComponent.h"

#include "Viewer/Rendering/ImGui/Layer.h"

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

namespace VkRender {
    /** Called once upon this object creation**/
    void PropertiesLayer::onAttach() {
    }

    /** Called after frame has finished rendered **/
    void PropertiesLayer::onFinishedRender() {
    }

    void PropertiesLayer::setScene(std::weak_ptr<Scene> scene) {
        Layer::setScene(scene);
        m_selectionContext = Entity(); // reset selectioncontext
    }

    bool PropertiesLayer::drawVec3Control(const std::string &label, glm::vec3 &values, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO &io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize)) {
            values.x = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &values.x, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Y", buttonSize)) {
            values.y = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##Y", &values.y, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Z", buttonSize)) {
            values.z = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##Z", &values.z, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();

        return valueChanged;
    }

    bool PropertiesLayer::drawVec2Control(const std::string &label, glm::vec2 &values, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO &io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize)) {
            values.x = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &values.x, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
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
        if (ImGui::DragFloat("##Y", &values.y, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();


        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();

        return valueChanged;
    }

    bool PropertiesLayer::drawFloatControl(const std::string &label, float &value, float resetValue = 0.0f,
                                           float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO &io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.5f, 0.5f, 0.5f, 1.0f}); // Gray
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.6f, 0.6f, 0.6f, 1.0f}); // Lighter gray
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.4f, 0.4f, 0.4f, 1.0f}); // Darker gray
        ImGui::PushFont(boldFont);
        if (ImGui::Button("R", buttonSize)) {
            value = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);

        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &value, 0.1f * speed, 0.0f, 0.0f, "%.2f")) {
            valueChanged = true;
        }
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();

        ImGui::Columns(1);

        ImGui::PopID();
        return valueChanged;
    }

    bool PropertiesLayer::drawQuatControl(const std::string &label, glm::quat &quat, float resetValue, float speed,
                                          float columnWidth) {
        bool valueChanged = false;
        ImGuiIO &io = ImGui::GetIO();
        auto boldFont = io.Fonts->Fonts[0];

        ImGui::PushID(label.c_str());

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, columnWidth);
        ImGui::Text("%s", label.c_str());
        ImGui::NextColumn();

        // Prepare space for 4 controls (W, X, Y, Z)
        ImGui::PushMultiItemsWidths(4, ImGui::CalcItemWidth());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});
        float fontSize = ImGui::GetFontSize();
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        float lineHeight = fontSize + framePadding.y * 2.0f;
        ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

        // --- Component W ---
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("W", buttonSize)) {
            quat.w = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        if (ImGui::DragFloat("##W", &quat.w, 0.1f * speed, 0.0f, 0.0f, "%.2f"))
            valueChanged = true;
        ImGui::PopItemWidth();
        ImGui::SameLine();

        // --- Component X ---
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("X", buttonSize)) {
            quat.x = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        if (ImGui::DragFloat("##X", &quat.x, 0.1f * speed, 0.0f, 0.0f, "%.2f"))
            valueChanged = true;
        ImGui::PopItemWidth();
        ImGui::SameLine();

        // --- Component Y ---
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Y", buttonSize)) {
            quat.y = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        if (ImGui::DragFloat("##Y", &quat.y, 0.1f * speed, 0.0f, 0.0f, "%.2f"))
            valueChanged = true;
        ImGui::PopItemWidth();
        ImGui::SameLine();

        // --- Component Z ---
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.9f, 0.9f, 0.2f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{1.0f, 1.0f, 0.3f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.9f, 0.9f, 0.2f, 1.0f});
        ImGui::PushFont(boldFont);
        if (ImGui::Button("Z", buttonSize)) {
            quat.z = resetValue;
            valueChanged = true;
        }
        ImGui::PopFont();
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        if (ImGui::DragFloat("##Z", &quat.z, 0.1f * speed, 0.0f, 0.0f, "%.2f"))
            valueChanged = true;
        ImGui::PopItemWidth();

        ImGui::PopStyleVar();
        ImGui::Columns(1);
        ImGui::PopID();

        return valueChanged;
    }


    template<typename T, typename UIFunction>
    void PropertiesLayer::drawComponent(const std::string &componentName, Entity entity, UIFunction uiFunction) {
        const ImGuiTreeNodeFlags treeNodeFlags =
                ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth |
                ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap;
        if (entity.hasComponent<T>()) {
            auto &component = entity.getComponent<T>();
            ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

            float fontSize = ImGui::GetFontSize();
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});

            float lineHeight = fontSize + 4.0f * 2.0f;
            ImGui::Separator();
            bool open = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), treeNodeFlags, "%s", componentName.c_str());
            ImGui::PopStyleVar();
            ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5f);

            if (ImGui::Button("+", ImVec2{lineHeight, lineHeight})) {
                ImGui::OpenPopup("ComponentSettings");
            }

            bool removeComponent = false;
            if (ImGui::BeginPopup("ComponentSettings")) {
                if (componentName != "Tag") {
                    if (ImGui::MenuItem("Remove component"))
                        removeComponent = true;
                }
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
            displayAddComponentEntry<TransformComponent>("Transform");
            displayAddComponentEntry<CameraComponent>("Camera");
            displayAddComponentEntry<MeshComponent>("Mesh");
            displayAddComponentEntry<MaterialComponent>("Material");
            displayAddComponentEntry<LightSourceComponent>("Light Source");
            displayAddComponentEntry<QuadricCollectionComponent>("Quadratic Collection");
            displayAddComponentEntry<ScriptableComponent>("Scriptable Component");

            ImGui::EndPopup();
        }

        drawComponent<TemporaryComponent>("TemporaryComponent", entity, [this](auto &component) {
        });


        drawComponent<TagComponent>("Tag", entity, [this](auto &component) {
            ImGui::Text("Entity Name:");
            ImGui::SameLine();
            // Define a buffer large enough to hold the tag's content
            // Copy the current tag content into the buffer
            // Check if `m_tagBuffer` is initialized or if the entity's tag has changed
            if (m_needsTagUpdate || strncmp(m_tagBuffer, component.getTag().c_str(), sizeof(m_tagBuffer)) != 0) {
                strncpy(m_tagBuffer, component.getTag().c_str(), sizeof(m_tagBuffer));
                m_tagBuffer[sizeof(m_tagBuffer) - 1] = '\0'; // Null-terminate to avoid overflow
                m_needsTagUpdate = false; // Reset the flag after updating the buffer
            }
            // Use ImGui::InputText to allow editing
            if (ImGui::InputText("##Tag", m_tagBuffer, sizeof(m_tagBuffer))) {
                // If the input changes, update the component's tag
                component.setTag(m_tagBuffer);
            }
        });

        drawComponent<TransformComponent>("Transform", entity, [this](TransformComponent &component) {
            bool paramsChanged = false;
            component.setMoving(paramsChanged);

            paramsChanged |= drawVec3Control("Translation", component.getPosition());
            glm::vec3 euler = component.rotationEuler;
            paramsChanged |= drawVec3Control("Rotation", euler, 0.0f);
            if (paramsChanged) {
                component.setRotationEuler(euler);
            }

            paramsChanged |= drawVec3Control("Scale", component.getScale(), 1.0f);

            if (paramsChanged) {
                component.setMoving(paramsChanged);
                component.updateFromEulerRotation();
            }
        });

        drawComponent<ScriptableComponent>("Scriptable", entity, [&entity](ScriptableComponent &component) {
            bool paramsChanged = false;
            std::vector<std::string> controllers = {
                "DefaultController",
                "Emitter",
                "SurfaceNormal",
                "ContributionRay",
                "GradientRay",
                "IntensityGradientRay",
            };

            static int selectedIndex = 0; // holds the currently selected controller index
            bool update = false;
            if (ImGui::BeginCombo("Controller", controllers[selectedIndex].c_str())) {
                for (int i = 0; i < controllers.size(); i++) {
                    bool isSelected = (selectedIndex == i);
                    if (ImGui::Selectable(controllers[i].c_str(), isSelected)) {
                        selectedIndex = i;
                        update = true;
                        if (entity.getComponent<ScriptableComponent>()) {
                            // unbind
                            auto &script = entity.getComponent<ScriptableComponent>();
                            script.destroyScript();
                        }
                    }
                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            if (!entity.getComponent<ScriptableComponent>() || update) {
                // When you need to bind the controller after selection:
                if (controllers[selectedIndex] == "DefaultController") {
                    entity.getComponent<ScriptableComponent>().bind<DefaultController>();
                } else if (controllers[selectedIndex] == "VectorScripts") {
                    entity.getComponent<ScriptableComponent>().bind<VectorScripts>();
                } else if (controllers[selectedIndex] == "Emitter") {
                    entity.getComponent<ScriptableComponent>().bind<Emitter>();
                } else if (controllers[selectedIndex] == "ContributionRay") {
                    entity.getComponent<ScriptableComponent>().bind<ContributionRay>();
                } else if (controllers[selectedIndex] == "SurfaceNormal") {
                    entity.getComponent<ScriptableComponent>().bind<SurfaceNormal>();
                } else if (controllers[selectedIndex] == "GradientRay") {
                    entity.getComponent<ScriptableComponent>().bind<GradientRay>();
                } else if (controllers[selectedIndex] == "IntensityGradientRay") {
                    entity.getComponent<ScriptableComponent>().bind<IntensityGradientRay>();
                }
            }
        });

        drawComponent<CameraComponent>("Camera", entity, [this, entity](CameraComponent &component) {
            //drawFloatControl("Field of View", component.camera->fov(), 1.0f);
            bool paramsChanged = false;

            paramsChanged |= ImGui::Checkbox("Render scene from viewpoint", &component.isActiveCamera());
            if (paramsChanged && component.isActiveCamera()) {
                // The user has just activated this camera.
                // Iterate over all camera components in the scene.
                auto view = m_context->activeScene()->getRegistry().view<CameraComponent>();
                for (auto entityID: view) {
                    Entity localEntity(entityID, m_context->activeScene().get());
                    // Skip the current entity (the one the user just toggled)
                    if (localEntity == entity)
                        continue;
                    auto &otherCameraComponent = localEntity.getComponent<CameraComponent>();
                    // Deactivate any camera that is not the current one.
                    if (otherCameraComponent.isActiveCamera()) {
                        otherCameraComponent.isActiveCamera() = false;
                    }
                }
            }

            paramsChanged |= ImGui::Checkbox("Flip Y", &component.cameraSettings.flipY);
            ImGui::SameLine();
            paramsChanged |= ImGui::Checkbox("Flip X", &component.cameraSettings.flipX);

            static const auto allCameraTypes = CameraComponent::getAllCameraTypes();
            static const auto cameraTypeStrings = []() {
                std::vector<std::string> strings;
                for (const auto &type: allCameraTypes) {
                    strings.push_back(CameraComponent::cameraTypeToString(type));
                }
                return strings;
            }();

            // Get the current camera type as a string
            std::string currentCameraTypeStr = CameraComponent::cameraTypeToString(component.cameraType);

            // Get the index of the current camera type
            int currentIndex = std::distance(
                cameraTypeStrings.begin(),
                std::find(cameraTypeStrings.begin(), cameraTypeStrings.end(), currentCameraTypeStr)
            );

            // ImGui combo box
            if (ImGui::BeginCombo("Camera Type", currentCameraTypeStr.c_str())) {
                for (int i = 0; i < cameraTypeStrings.size(); ++i) {
                    bool isSelected = (i == currentIndex);
                    if (ImGui::Selectable(cameraTypeStrings[i].c_str(), isSelected)) {
                        currentIndex = i;
                        component.cameraType = allCameraTypes[i];
                        paramsChanged |= true;
                    }

                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }


            switch (component.cameraType) {
                case CameraComponent::PERSPECTIVE:
                    paramsChanged |= ImGui::SliderFloat("Field of View", &component.baseCameraParameters.fov, 5.0f,
                                                        180.0f);
                    paramsChanged |= ImGui::SliderFloat("Aspect Ratio", &component.baseCameraParameters.aspect, 0.1f,
                                                        10.0f);
                    paramsChanged |= ImGui::SliderFloat("Near Plane", &component.baseCameraParameters.nearPlane, 0.01f,
                                                        10.0f);
                    paramsChanged |= ImGui::SliderFloat("Far Plane", &component.baseCameraParameters.farPlane, 1.0f,
                                                        1000.0f);

                    break;
                case CameraComponent::PINHOLE:
                    paramsChanged |= ImGui::SliderFloat("Width", &component.pinholeParameters.width, 1.0, 4096, "%.0f");
                    paramsChanged |= ImGui::SliderFloat("Height", &component.pinholeParameters.height, 1.0f, 4096.0f,
                                                        "%.0f");

                    paramsChanged |= ImGui::SliderFloat("Fx", &component.pinholeParameters.fx, 1.0f, 4096.0f);
                    paramsChanged |= ImGui::SliderFloat("Fy", &component.pinholeParameters.fy, 1.0f, 4096.0f);
                    paramsChanged |= ImGui::SliderFloat("Cx", &component.pinholeParameters.cx, 1.0f, 4096.0f);
                    paramsChanged |= ImGui::SliderFloat("Cy", &component.pinholeParameters.cy, 1.0f, 4096.0f);
                    paramsChanged |= ImGui::SliderFloat("Focal Length", &component.pinholeParameters.focalLength, 1.0f,
                                                        100.0f);
                    paramsChanged |= ImGui::SliderFloat("Aperture", &component.pinholeParameters.fNumber, 0.0f, 32.0f);

                    if (ImGui::Button("Set from viewport")) {
                        auto &ci = m_context->getViewport()->getCreateInfo();
                        component.pinholeParameters.width = ci.width;
                        component.pinholeParameters.height = ci.height;
                        component.pinholeParameters.cx = ci.width / 2;
                        component.pinholeParameters.cy = ci.height / 2;
                        paramsChanged = true;
                    }
                    break;
                case CameraComponent::ARCBALL:
                    break;
            }

            component.resetUpdateState();
            // TODO not a nice way of ensuring single frame updates from camera properties. Can we make this less error-prone? Relying on paramsChanged variable and  m_updateTrigger in cameraComponent struct
            if (paramsChanged) {
                component.updateParametersChanged();
            }
            component.camera->updateProjectionMatrix();
        });

        drawComponent<MeshComponent>("Mesh", entity, [this, &entity](MeshComponent &component) {
            // Mesh Type Selection
            int currentMeshType = component.meshDataType();
            // Create the combo and check for interaction
            // Polygon Mode Control
            ImGui::Text("Polygon Mode:");
            const char *polygonModes[] = {"Line", "Fill"};
            int currentMode = (component.polygonMode() == VK_POLYGON_MODE_LINE) ? 0 : 1;
            if (ImGui::Combo("Polygon Mode", &currentMode, polygonModes, IM_ARRAYSIZE(polygonModes))) {
                if (currentMode == 0) {
                    component.polygonMode() = VK_POLYGON_MODE_LINE;
                } else {
                    component.polygonMode() = VK_POLYGON_MODE_FILL;
                }
                m_context->activeScene()->onComponentUpdated(entity, component);
            }

            // Begin the combo box
            if (ImGui::BeginCombo("Mesh Type",
                                  meshDataTypeToString(static_cast<MeshDataType>(currentMeshType)).c_str())) {
                // Loop through the available mesh types
                for (size_t i = 0; i < meshDataTypeToArray().size(); ++i) {
                    bool isSelected = (currentMeshType == static_cast<int>(meshDataTypeToArray()[i]));
                    if (ImGui::Selectable(meshDataTypeToString(meshDataTypeToArray()[i]).c_str(), isSelected)) {
                        currentMeshType = static_cast<int>(meshDataTypeToArray()[i]);
                        component.meshDataType() = static_cast<MeshDataType>(currentMeshType);

                        // Trigger behavior when a new type is selected
                        switch (component.meshDataType()) {
                            case MeshDataType::OBJ_FILE:

                                EditorUtils::openImportFileDialog("Wavefront", {".obj"}, LayerUtils::OBJ_FILE,
                                                                  &m_loadFileFuture);
                                break;

                            case MeshDataType::PLY_FILE:

                                EditorUtils::openImportFileDialog("Stanford .PLY", {".ply"}, LayerUtils::PLY_MESH,
                                                                  &m_loadFileFuture);
                                break;
                            case MeshDataType::CYLINDER:
                                component.meshParameters = std::make_shared<CylinderMeshParameters>();
                                break;
                            case MeshDataType::QUADRIC:
                                component.meshParameters = std::make_shared<QuadricMeshParameters>();
                                break;
                            case MeshDataType::CAMERA_GIZMO_PINHOLE:
                                component.meshParameters = std::make_shared<CameraGizmoPinholeMeshParameters>();
                                break;
                            case MeshDataType::CAMERA_GIZMO_PERSPECTIVE:
                                component.meshParameters = std::make_shared<CameraGizmoPerspectiveMeshParameters>();
                                break;
                            case MeshDataType::CUBE:
                                component.meshParameters = std::make_shared<CubeMeshParameters>();
                                break;
                            case MeshDataType::PLANE:
                                component.meshParameters = std::make_shared<PlaneMeshParameters>();
                                break;
                            default:
                                Log::Logger::getInstance()->error("Unknown mesh type!");
                                break;
                        }
                    }

                    // Ensure the currently selected item remains selected
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }

                ImGui::EndCombo();
            }

            // Display different input fields based on mesh type
            switch (component.meshDataType()) {
                case OBJ_FILE: {
                    auto params = std::dynamic_pointer_cast<OBJFileMeshParameters>(component.meshParameters);
                    if (params) {
                        ImGui::Text("Mesh File:");
                        ImGui::Text("%s", params->path.empty() ? "" : params->path.string().c_str());
                    }
                }
                break;
                case PLY_FILE: {
                    auto params = std::dynamic_pointer_cast<PLYFileMeshParameters>(component.meshParameters);
                    if (params) {
                        ImGui::Text("Mesh File:");
                        ImGui::Text("%s", params->path.empty() ? "" : params->path.string().c_str());
                    }
                }
                break;

                case MeshDataType::CYLINDER: {
                    auto cylinderParams = std::dynamic_pointer_cast<CylinderMeshParameters>(component.meshParameters);
                    if (cylinderParams) {
                        bool paramsChanged = false;
                        paramsChanged |= drawVec3Control("Origin", cylinderParams->origin);
                        paramsChanged |= drawVec3Control("Direction", cylinderParams->direction, 0.0f, 0.05f);
                        paramsChanged |= ImGui::SliderFloat("Magnitude", &cylinderParams->magnitude, 0.0f, 100.0f);
                        paramsChanged |= ImGui::SliderFloat("Radius", &cylinderParams->radius, 0.001f, 0.1f);
                        if (paramsChanged) {
                            cylinderParams->setDirty();
                        }
                    }
                    break;
                }
                case MeshDataType::QUADRIC: {
                    auto quadricParams = std::dynamic_pointer_cast<QuadricMeshParameters>(component.meshParameters);
                    if (quadricParams) {
                        bool paramsChanged = false;
                        paramsChanged |= ImGui::SliderInt("GridResolution", &quadricParams->gridResolution, 0.0f,
                                                          10000.0f);
                        paramsChanged |= drawVec2Control("Min", quadricParams->min);
                        paramsChanged |= drawVec2Control("Max", quadricParams->max);

                        paramsChanged |= ImGui::SliderFloat("tx", &quadricParams->t_x, -5.0f, 5.0f);
                        paramsChanged |= ImGui::SliderFloat("ty", &quadricParams->t_y, -5.0f, 5.0f);
                        paramsChanged |= ImGui::SliderFloat("a", &quadricParams->a, -5.0f, 5.0f);
                        paramsChanged |= ImGui::SliderFloat("b", &quadricParams->b, -5.0f, 5.0f);
                        paramsChanged |= ImGui::SliderFloat("c", &quadricParams->c, -5.0f, 5.0f);
                        ImGui::Separator();
                        ImGui::Text("Beta Kernel opts");
                        paramsChanged |= ImGui::SliderFloat("b_beta", &quadricParams->b_beta, -5.0f, 5.0f);
                        paramsChanged |= ImGui::SliderFloat("threshold", &quadricParams->threshold, 0.0f, 1.0f);
                        paramsChanged |= ImGui::SliderFloat("scale", &quadricParams->kernelScale, 0.0f, 10.0f);
                        paramsChanged |= ImGui::SliderFloat("circularity", &quadricParams->circularity, 0.0f, 1.0f);
                        if (paramsChanged) {
                            quadricParams->setDirty();
                        }
                    }
                    break;
                }
                    ImGui::Dummy(ImVec2(5.0f, 5.0f));

                case MeshDataType::CAMERA_GIZMO_PINHOLE: {
                    if (entity.hasComponent<CameraComponent>()) {
                        auto cameraGizmoParams = std::dynamic_pointer_cast<CameraGizmoPinholeMeshParameters>(
                            component.meshParameters);
                        if (cameraGizmoParams) {
                            auto &cameraParams = entity.getComponent<CameraComponent>().pinholeParameters;
                            // Update focal point and check if it has changed
                            if (cameraGizmoParams->parameters != cameraParams) {
                                cameraGizmoParams->parameters = cameraParams;
                                cameraGizmoParams->setDirty();
                            }
                        }
                    } else {
                        if (ImGui::Button("Add a camera component!")) {
                            entity.addComponent<CameraComponent>();
                        }
                    }
                    break;

                case MeshDataType::CAMERA_GIZMO_PERSPECTIVE: {
                    if (entity.hasComponent<CameraComponent>()) {
                        auto cameraGizmoParams = std::dynamic_pointer_cast<CameraGizmoPerspectiveMeshParameters>(
                            component.meshParameters);
                        if (cameraGizmoParams) {
                            auto &cameraParams = entity.getComponent<CameraComponent>().baseCameraParameters;
                            // Update focal point and check if it has changed
                            if (cameraGizmoParams->parameters != cameraParams) {
                                cameraGizmoParams->parameters = cameraParams;
                                cameraGizmoParams->setDirty();
                            }
                        }
                    } else {
                        if (ImGui::Button("Add a camera component!")) {
                            entity.addComponent<CameraComponent>();
                        }
                    }
                }
                break;
                }
                default:
                    break;
            }
        });


        drawComponent<MaterialComponent>("Material", entity, [this, entity](MaterialComponent &component) {
            ImGui::Text("Material Properties");

            // Base Color Control
            ImGui::Text("Base Color");
            ImGui::ColorEdit4("##BaseColor", glm::value_ptr(component.albedo));

            ImGui::Text("Appearance Properties");
            bool update = false;
            update |= drawFloatControl("Emission", component.emission, 0.0f, 0.1f);
            update |= drawFloatControl("Diffuse", component.diffuse, 0.5f, 0.1f);
            update |= drawFloatControl("Specular", component.specular, 0.5f, 0.1f);
            update |= drawFloatControl("PhongExponents", component.phongExponent, 32.0f, 1.0f);

            /*
            // Emissive Factor Control
            ImGui::Text("Emissive Factor");
            ImGui::ColorEdit3("##EmissiveFactor", glm::value_ptr(component.emissiveFactor));
*/

            if (ImGui::Button("Reload Material Shader")) {
                component.reloadShader = true;
                m_context->activeScene()->onComponentUpdated(entity, component);
            }

            ImGui::Checkbox("Apply Texture", &component.useTexture);

            ImGui::Dummy(ImVec2(5.0f, 5.0f));
            ImGui::PushFont(m_editor->guiResources().font15);
            ImGui::Text("Texture");
            ImGui::PopFont();

            ImGui::Text("Source:");
            ImGui::Text("%s", component.albedoTexturePath.string().c_str());
            // Button to load texture
            if (ImGui::Button("Set Texture Image")) {
                std::vector<std::string> types{".png", ".jpg", ".bmp"};
                EditorUtils::openImportFileDialog("Load Texture", types, LayerUtils::TEXTURE_FILE, &m_loadFileFuture);
            }


            // Shader Controls
            ImGui::Text("Vertex Shader:");
            ImGui::Text("%s", component.vertexShaderName.string().c_str());
            if (ImGui::Button("Load Vertex Shader")) {
                std::vector<std::string> types{".vert"};
                EditorUtils::openImportFileDialog("Load Vertex Shader", types, LayerUtils::VERTEX_SHADER_FILE,
                                                  &m_loadFileFuture);
            }

            ImGui::Text("Fragment Shader:");
            ImGui::Text("%s", component.fragmentShaderName.string().c_str());
            if (ImGui::Button("Load Fragment Shader")) {
                std::vector<std::string> types{".frag"};
                EditorUtils::openImportFileDialog("Load Fragment Shader", types, LayerUtils::FRAGMENT_SHADER_FILE,
                                                  &m_loadFileFuture);
            }
            // Notify scene that material component has been updated
        });

        drawComponent<LightSourceComponent>("Gaussian Model", entity, [this](LightSourceComponent &component) {
            ImGui::Text("Gaussian Model Properties");

            // Display the number of Gaussians
            size_t gaussianCount = component.size();
            ImGui::Text("Number of Gaussians: %zu", gaussianCount);

            ImGui::Separator();

            // Button to add a new Gaussian
            if (ImGui::Button("Add Gaussian")) {
                // Default values for a new Gaussian
                glm::vec3 defaultMean(0.0f, 0.0f, 0.0f);
                glm::vec3 defaultNormal(0.0f, 0.0f, 1.0f); // Identity matrix
                glm::vec2 defaultScale(1.0f); // Identity matrix
                component.addGaussian(defaultMean, defaultNormal, defaultScale);
            }

            ImGui::SameLine();

            if (ImGui::Button("Load from file")) {
                std::vector<std::string> types{".ply"};
                EditorUtils::openImportFileDialog("Load 3DGS .ply file", types, LayerUtils::PLY_3DGS,
                                                  &m_loadFileFuture);
            }

            if (ImGui::Button("Remove All")) {
                component.removeAllGaussians();
            }
            ImGui::Spacing();


            // Iterate over each Gaussian and provide controls to modify them
            if (component.size() < 10) {
                for (size_t i = 0; i < component.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i)); // Ensure unique ID for ImGui widgets
                    // Collapsible header for each Gaussian
                    if (ImGui::CollapsingHeader(("Gaussian " + std::to_string(i)).c_str())) {
                        // Mean Position Controls
                        bool update = false;
                        update |= drawVec3Control("Position", component.positions[i], 0.0f);
                        update |= drawVec3Control("Normal", component.normals[i], 0.0f, 0.1f);
                        update |= drawVec2Control("Scale", component.scales[i], 0.0f, 0.1f);
                        // Amplitude Control
                        ImGui::Text("Appearance Properties");
                        update |= drawFloatControl("Opacity", component.opacities[i], 0.0f, 0.1f);
                        update |= drawFloatControl("Emission", component.emissions[i], 0.0f, 0.1f);
                        update |= drawFloatControl("Diffuse", component.diffuse[i], 0.5f, 0.1f);
                        update |= drawFloatControl("Specular", component.specular[i], 0.5f, 0.1f);
                        update |= drawFloatControl("PhongExponents", component.phongExponents[i], 32.0f, 1.0f);

                        ImGui::Spacing();
                        ImGui::PushItemWidth(200); // Set a wider width for the next widget
                        update |= ImGui::ColorEdit4("Color", glm::value_ptr(component.colors[i]));
                        ImGui::PopItemWidth(); // Revert to the previous width
                        // Button to remove this Gaussian
                        ImGui::Spacing();
                        if (ImGui::Button("Remove Gaussian")) {
                            component.positions.erase(component.positions.begin() + i);
                            component.scales.erase(component.scales.begin() + i);
                            component.normals.erase(component.normals.begin() + i);
                            component.emissions.erase(component.emissions.begin() + i);
                            component.opacities.erase(component.opacities.begin() + i);
                            component.colors.erase(component.colors.begin() + i);
                            component.diffuse.erase(component.diffuse.begin() + i);
                            component.specular.erase(component.specular.begin() + i);
                            component.phongExponents.erase(component.phongExponents.begin() + i);
                            --i; // Adjust index after removal
                        }
                    }

                    ImGui::PopID(); // Pop ID for this Gaussian
                }
                return;
            }
            // For large numbers of Gaussians, show a single "selected" Gaussian
            // ----------------------------------------------------------------

            static int selectedGaussianIndex = 0; // or persist somewhere, e.g. as a class member

            // Ensure valid range
            if (selectedGaussianIndex < 0) selectedGaussianIndex = 0;
            if (selectedGaussianIndex >= (int) gaussianCount) {
                selectedGaussianIndex = (int) gaussianCount - 1;
            }

            // UI to pick which Gaussian to inspect
            ImGui::Text("Edit a Single Gaussian (Large Set)");
            ImGui::PushItemWidth(120.0f);
            ImGui::InputInt("Gaussian Index", &selectedGaussianIndex);
            ImGui::PopItemWidth();

            // Clamp again after user input
            if (selectedGaussianIndex < 0) selectedGaussianIndex = 0;
            if (selectedGaussianIndex >= (int) gaussianCount) {
                selectedGaussianIndex = (int) gaussianCount - 1;
            }

            // Navigation buttons to move up/down
            ImGui::SameLine();
            if (ImGui::ArrowButton("PrevGaussian", ImGuiDir_Left)) {
                selectedGaussianIndex--;
                if (selectedGaussianIndex < 0) selectedGaussianIndex = 0;
            }
            ImGui::SameLine();
            if (ImGui::ArrowButton("NextGaussian", ImGuiDir_Right)) {
                selectedGaussianIndex++;
                if (selectedGaussianIndex >= (int) gaussianCount) {
                    selectedGaussianIndex = (int) gaussianCount - 1;
                }
            }

            ImGui::Separator();

            // Now display and edit ONLY the selected Gaussian
            {
                size_t i = (size_t) selectedGaussianIndex;

                ImGui::Text("Selected Gaussian %d", selectedGaussianIndex + 1);

                bool update = false;
                update |= drawVec3Control("Position", component.positions[i], 0.0f);
                update |= drawVec3Control("Normal", component.normals[i], 0.0f, 0.1f);
                update |= drawVec2Control("Scale", component.scales[i], 0.0f, 0.1f);

                ImGui::Text("Appearance Properties");
                update |= drawFloatControl("Opacity", component.opacities[i], 0.0f, 0.1f);
                update |= drawFloatControl("Emission", component.emissions[i], 0.0f, 0.1f);
                update |= drawFloatControl("Diffuse", component.diffuse[i], 0.5f, 0.1f);
                update |= drawFloatControl("Specular", component.specular[i], 0.5f, 0.1f);
                update |= drawFloatControl("PhongExp", component.phongExponents[i], 32.0f, 1.0f);

                update |= ImGui::ColorEdit4("Color", glm::value_ptr(component.colors[i]));

                ImGui::Spacing();

                if (ImGui::Button("Remove This Gaussian")) {
                    component.positions.erase(component.positions.begin() + i);
                    component.normals.erase(component.normals.begin() + i);
                    component.scales.erase(component.scales.begin() + i);
                    component.emissions.erase(component.emissions.begin() + i);
                    component.colors.erase(component.colors.begin() + i);
                    component.diffuse.erase(component.diffuse.begin() + i);
                    component.specular.erase(component.specular.begin() + i);
                    component.phongExponents.erase(component.phongExponents.begin() + i);

                    // Adjust if we removed the last one
                    if (i >= component.size()) {
                        i = component.size() - 1;
                    }
                    selectedGaussianIndex = (int) i;
                }
            }
        });

        drawComponent<VkRender::QuadricCollectionComponent>(
            "Quadratic Model", entity,
            [this, &entity](
        VkRender::QuadricCollectionComponent &component) {
                if (!entity.hasComponent<MeshComponent>()) {
                    entity.addComponent<MeshComponent>();
                }
                if (!entity.hasComponent<GroupComponent>()) {
                    entity.addComponent<GroupComponent>();
                }

                auto &modelTransform = entity.getOrAddComponent<
                    TransformComponent>();

                ImGui::Text("Quadratic Model Properties");

                // Display the number of quadrics
                size_t quadricCount = component.size();
                ImGui::Text("Number of Quadrics: %zu", quadricCount);
                ImGui::Separator();

                // Button to add a new Quadric with default values
                if (ImGui::Button("Add Quadric")) {
                    glm::vec3 defaultPos(0.0f, 0.0f, 0.0f);
                    glm::quat defaultRot(1.0f, 0.0f, 0.0f, 0.0f);
                    // Identity rotation
                    // Default shape parameters: a, b, c, t_x, t_y.
                    // For example, a and b = 1, c = 1 (curvature), t_x and t_y = 2 to push tanh toward 1.
                    float defaultA = 1.0f;
                    float defaultB = 1.0f;
                    float defaultC = 1.0f;
                    float defaultTx = -1.0f;
                    float defaultTy = 1.0f;
                    // Additional constants: kernelScale = 1, threshold = 0.01, beta = 0.
                    component.addQuadric(
                        defaultPos, defaultRot, defaultA, defaultB,
                        defaultC, defaultTx, defaultTy, 1.0f, 0.01f,
                        0.0f);
                }
                ImGui::SameLine();
                ImGui::Checkbox("Add Noise", &m_tmp);
                ImGui::SameLine();
                ImGui::Checkbox("Visibility", &m_visibility);
                ImGui::SameLine();

                if (ImGui::Button("Load from file")) {
                    std::vector<std::string> types{".ply"};
                    EditorUtils::openImportFileDialog(
                        "Load Quadratic .ply file", types,
                        LayerUtils::PLY_QUADRATIC, &m_loadFileFuture);
                }

                ImGui::SliderFloat("Noise", &noiseSlider, 0.0f, 3.0f);

                if (ImGui::Button("Remove All")) {
                    for (int i = 0; i < component.size(); ++i) {
                        std::string quadricName =
                                "Quadric " + std::to_string(i);
                        auto entityInstance = m_context->activeScene()->
                                getOrCreateEntityByName(quadricName);
                        m_context->activeScene()->
                                destroyEntityRecursively(
                                    entityInstance);
                    }
                    component.removeAllQuadrics();
                    quadricCount = component.size();
                }

                ImGui::SameLine();
                if (ImGui::Button("Update Transforms")) {
                    for (int i = 0; i < component.size(); ++i) {
                        // TODO loop over parent entity' children instead of this
                        std::string quadricName =
                                "Quadric " + std::to_string(i);
                        auto entityInstance = m_context->activeScene()->
                                getOrCreateEntityByName(quadricName);
                        entityInstance.setParent(entity);
                        // TODO verify that the selected entity is indeed the quadric collection
                        auto &transform = entityInstance.
                                getOrAddComponent<TransformComponent>();
                        transform.setPosition(component.positions[i]);
                        transform.setRotationQuaternion(
                            component.rotations[i]);
                        glm::mat4 parentMatrix = entity.getComponent<
                            TransformComponent>().getTransform();
                        // Get parent's transformation matrix
                        glm::mat4 worldMatrix =
                                parentMatrix * transform.getTransform();
                        transform.setTransform(worldMatrix);
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Add Positional Noise")) {
                    for (int i = 0; i < component.size(); ++i) {
                        std::string quadricName =
                                "Quadric " + std::to_string(i);
                        auto position = component.positions[i];

                        std::random_device rd;
                        std::mt19937 gen(rd()); // Mersenne Twister RNG
                        std::normal_distribution<float> dist(0.0f, noiseSlider); // Mean 0, standard deviation 0.01

                        position.x += dist(gen);
                        position.y += dist(gen);
                        position.z += dist(gen);
                        component.positions[i] = position;
                    }
                }

                ImGui::Spacing();

                // For a small number of quadrics, display all entries
                if (quadricCount < 10) {
                    for (size_t i = 0; i < quadricCount; ++i) {
                        ImGui::PushID(static_cast<int>(i));
                        // Unique ID for ImGui widgets
                        std::string quadricName =
                                "Quadric " + std::to_string(i);
                        auto entityInstance = m_context->activeScene()->
                                getOrCreateEntityByName(quadricName);
                        entityInstance.setParent(entity);
                        auto &transform = entityInstance.
                                getOrAddComponent<TransformComponent>();
                        transform.setPosition(component.positions[i]);
                        transform.setRotationQuaternion(
                            component.rotations[i]);
                        glm::mat4 parentMatrix = modelTransform.
                                getTransform();
                        // Get parent's transformation matrix
                        glm::mat4 worldMatrix =
                                parentMatrix * transform.getTransform();
                        transform.setTransform(worldMatrix);

                        auto &mesh = entityInstance.getOrAddComponent<
                            MeshComponent>(QUADRIC);
                        auto quadricParams = std::dynamic_pointer_cast<
                            QuadricMeshParameters>(mesh.meshParameters);

                        auto &material = entityInstance.
                                getOrAddComponent<MaterialComponent>();
                        material.useTexture = true;

                        if (ImGui::CollapsingHeader(
                            (quadricName).c_str())) {
                            bool update = false;
                            update |= drawVec3Control(
                                "Position", component.positions[i],
                                0.0f);

                            glm::vec3 euler = glm::eulerAngles(component.rotations[i]);
                            bool updated = drawVec3Control("Rotation", euler, 0.0f);
                            if (updated) {
                                component.rotations[i] = glm::quat(euler);
                                update |= true;
                            }

                            update |= drawFloatControl(
                                "a", component.a[i], 1.0f, 0.1f);
                            update |= drawFloatControl(
                                "b", component.b[i], 1.0f, 0.1f);
                            update |= drawFloatControl(
                                "c (Curvature)", component.c[i], 1.0f,
                                0.1f);
                            update |= drawFloatControl(
                                "t_x", component.t_x[i], 2.0f, 0.1f);
                            update |= drawFloatControl(
                                "t_y", component.t_y[i], 2.0f, 0.1f);
                            update |= drawFloatControl(
                                "Kernel Scale",
                                component.kernelScale[i], 1.0f, 0.1f);
                            update |= drawFloatControl(
                                "Threshold", component.threshold[i],
                                0.01f, 0.001f);
                            update |= drawFloatControl(
                                "Beta", component.beta[i], 0.0f, 0.1f);
                            ImGui::Separator();
                            update |= ImGui::SliderInt(
                                "GridResolution",
                                &quadricParams->gridResolution, 0.0f,
                                1000.0f);


                            ImGui::Spacing();
                            if (ImGui::Button("Remove Quadric")) {
                                m_context->activeScene()->
                                        destroyEntityRecursively(
                                            entityInstance);
                                component.positions.erase(
                                    component.positions.begin() + i);
                                component.rotations.erase(
                                    component.rotations.begin() + i);
                                component.a.erase(
                                    component.a.begin() + i);
                                component.b.erase(
                                    component.b.begin() + i);
                                component.c.erase(
                                    component.c.begin() + i);
                                component.t_x.erase(
                                    component.t_x.begin() + i);
                                component.t_y.erase(
                                    component.t_y.begin() + i);
                                component.kernelScale.erase(
                                    component.kernelScale.begin() + i);
                                component.threshold.erase(
                                    component.threshold.begin() + i);
                                component.beta.erase(
                                    component.beta.begin() + i);
                                --quadricCount;
                                --i; // Adjust index after removal
                            }

                            if (update) {
                                quadricParams->setDirty();
                                quadricParams->a = component.a[i];
                                quadricParams->b = component.b[i];
                                quadricParams->c = component.c[i];
                                quadricParams->t_x = component.t_x[i];
                                quadricParams->t_y = component.t_y[i];
                                quadricParams->kernelScale = component.
                                        kernelScale[i];
                                quadricParams->threshold = component.
                                        threshold[i];
                                quadricParams->b_beta = component.beta[
                                    i];
                            }
                        }


                        ImGui::PopID();
                    }

                    return;
                }

                // For a large number of quadrics, show a single "selected" quadric for editing.
                static int selectedQuadricIndex = 0;
                if (selectedQuadricIndex < 0)
                    selectedQuadricIndex = 0;
                if (selectedQuadricIndex >= (int) quadricCount)
                    selectedQuadricIndex = (int) quadricCount - 1;

                ImGui::Text("Edit a Single Quadric (Large Set)");
                ImGui::PushItemWidth(120.0f);
                ImGui::InputInt("Quadric Index", &selectedQuadricIndex);
                ImGui::PopItemWidth();
                if (selectedQuadricIndex < 0)
                    selectedQuadricIndex = 0;
                if (selectedQuadricIndex >= (int) quadricCount)
                    selectedQuadricIndex = (int) quadricCount - 1;

                ImGui::SameLine();
                if (ImGui::ArrowButton("PrevQuadric", ImGuiDir_Left)) {
                    selectedQuadricIndex--;
                    if (selectedQuadricIndex < 0)
                        selectedQuadricIndex = 0;
                }
                ImGui::SameLine();
                if (ImGui::ArrowButton("NextQuadric", ImGuiDir_Right)) {
                    selectedQuadricIndex++;
                    if (selectedQuadricIndex >= (int) quadricCount)
                        selectedQuadricIndex = (int) quadricCount - 1;
                }
                ImGui::Separator();
                {
                    size_t i = static_cast<size_t>(
                        selectedQuadricIndex);

                    std::string quadricName =
                            "Quadric " + std::to_string(
                                selectedQuadricIndex);
                    auto entityInstance = m_context->activeScene()->
                            getOrCreateEntityByName(quadricName);
                    entityInstance.setParent(entity);
                    auto &transform = entityInstance.getOrAddComponent<
                        TransformComponent>();
                    transform.setPosition(component.positions[i]);
                    transform.setRotationQuaternion(
                        component.rotations[i]);
                    glm::mat4 parentMatrix = modelTransform.
                            getTransform();
                    // Get parent's transformation matrix
                    glm::mat4 worldMatrix =
                            parentMatrix * transform.getTransform();
                    transform.setTransform(worldMatrix);
                    auto &mesh = entityInstance.getOrAddComponent<
                        MeshComponent>(QUADRIC);
                    auto quadricParams = std::dynamic_pointer_cast<
                        QuadricMeshParameters>(mesh.meshParameters);
                    auto &material = entityInstance.getOrAddComponent<
                        MaterialComponent>();
                    material.useTexture = true;


                    ImGui::Text(
                        "Selected Quadric %d",
                        selectedQuadricIndex + 1);
                    bool update = false;
                    update |= drawVec3Control(
                        "Position", component.positions[i], 0.0f);
                    update |= drawQuatControl(
                        "Rotation", component.rotations[i]);
                    update |= drawFloatControl(
                        "a", component.a[i], 1.0f, 0.1f);
                    update |= drawFloatControl(
                        "b", component.b[i], 1.0f, 0.1f);
                    update |= drawFloatControl(
                        "c (Curvature)", component.c[i], 1.0f, 0.1f);
                    update |= drawFloatControl(
                        "t_x", component.t_x[i], 2.0f, 0.1f);
                    update |= drawFloatControl(
                        "t_y", component.t_y[i], 2.0f, 0.1f);
                    update |= drawFloatControl(
                        "Kernel Scale", component.kernelScale[i], 1.0f,
                        0.1f);
                    update |= drawFloatControl(
                        "Threshold", component.threshold[i], 0.01f,
                        0.001f);
                    update |= drawFloatControl(
                        "Beta", component.beta[i], 0.0f, 0.1f);
                    ImGui::Spacing();

                    if (ImGui::Button("Remove This Quadric")) {
                        m_context->activeScene()->
                                destroyEntityRecursively(
                                    entityInstance);

                        component.positions.erase(
                            component.positions.begin() + i);
                        component.rotations.erase(
                            component.rotations.begin() + i);
                        component.a.erase(component.a.begin() + i);
                        component.b.erase(component.b.begin() + i);
                        component.c.erase(component.c.begin() + i);
                        component.t_x.erase(component.t_x.begin() + i);
                        component.t_y.erase(component.t_y.begin() + i);
                        component.kernelScale.erase(
                            component.kernelScale.begin() + i);
                        component.threshold.erase(
                            component.threshold.begin() + i);
                        component.beta.
                                erase(component.beta.begin() + i);
                        if (i >= component.size() && component.size() >
                            0)
                            i = component.size() - 1;
                        selectedQuadricIndex = static_cast<int>(i);
                    }

                    if (update) {
                        quadricParams->setDirty();
                        quadricParams->a = component.a[i];
                        quadricParams->b = component.b[i];
                        quadricParams->c = component.c[i];
                        quadricParams->t_x = component.t_x[i];
                        quadricParams->t_y = component.t_y[i];
                        quadricParams->kernelScale = component.
                                kernelScale[i];
                        quadricParams->threshold = component.threshold[
                            i];
                        quadricParams->b_beta = component.beta[i];
                    }
                }
            });


        drawComponent<GroupComponent>("Group", entity, [this](auto &component) {
        });
    }


    /** Called once per frame **/
    void PropertiesLayer::onUIRender() {
        m_selectionContext = m_context->getSelectedEntity();
        ImVec2 window_pos = ImVec2(0.0f, m_editor->ui()->layoutConstants.uiYOffset); // Position (x, y)
        ImVec2 window_size = ImVec2(m_editor->ui()->width, m_editor->ui()->height - window_pos.y);
        // Size (width, height)
        // Set window flags to remove decorations
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoBringToFrontOnFocus;

        // Set next window position and size
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

        // Create the parent window
        ImGui::Begin("PropertiesLayer", nullptr, window_flags);

        ImGui::Text("Entity Properties");
        std::shared_ptr<Scene> scene = m_context->activeScene();
        if (ImGui::Button("Delete This Entity")) {
            scene->destroyEntityRecursively(m_context->getSelectedEntity());
        }
        ImGui::SameLine();

        if (m_selectionContext) {
            drawComponents(m_selectionContext);
        }

        checkFileImportCompletion();
        checkFolderImportCompletion();

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
    PropertiesLayer::handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo &loadFileInfo) {
        if (!loadFileInfo.path.empty()) {
            switch (loadFileInfo.filetype) {
                case LayerUtils::TEXTURE_FILE: {
                    auto &materialComponent = m_selectionContext.getComponent<MaterialComponent>();
                    materialComponent.albedoTexturePath = loadFileInfo.path;
                    m_context->activeScene()->onComponentUpdated(m_selectionContext, materialComponent);
                }
                break;
                case LayerUtils::OBJ_FILE:
                    // Load into the active scene
                    if (m_selectionContext.hasComponent<MeshComponent>()) {
                        auto &meshComponent = m_selectionContext.getComponent<MeshComponent>();
                        auto param = std::dynamic_pointer_cast<OBJFileMeshParameters>(meshComponent.meshParameters);
                        if (param) {
                            param->path = loadFileInfo.path;
                            param->setDirty();
                        } else {
                            m_selectionContext.removeComponent<MeshComponent>();
                            auto &meshComponent = m_selectionContext.addComponent<MeshComponent>(
                                OBJ_FILE, loadFileInfo.path);
                        }
                    }

                    break;
                case LayerUtils::PLY_3DGS: {
                    if (m_selectionContext.hasComponent<LightSourceComponent>()) {
                        auto &comp = m_selectionContext.getComponent<LightSourceComponent>();
                        comp.addGaussiansFromFile(loadFileInfo.path);
                    }
                }
                break;
                case LayerUtils::PLY_QUADRATIC: {
                    if (m_selectionContext.hasComponent<QuadricCollectionComponent>()) {
                        auto &comp = m_selectionContext.getComponent<QuadricCollectionComponent>();
                        comp.addQuadricsFromFile(loadFileInfo.path, m_tmp, noiseSlider);

                        // Now add quadrics to scene:

                        int numEntities = comp.size();
                        // Compute step such that we do not exceed 200 entities.
                        auto &visibility = m_selectionContext.getOrAddComponent<VisibleComponent>();
                        visibility.visible = m_visibility;

                        for (int i = 0; i < numEntities; ++i) {
                            std::string quadricName = "Quadric " + std::to_string(i);
                            auto entityInstance = m_context->activeScene()->getOrCreateEntityByName(quadricName);
                            entityInstance.setParent(m_selectionContext);

                            // Get or create TransformComponent and set position and rotation.
                            auto &transform = entityInstance.getOrAddComponent<TransformComponent>();
                            transform.setPosition(comp.positions[i]);
                            transform.setRotationQuaternion(comp.rotations[i]);

                            // Apply parent's transformation.
                            glm::mat4 parentMatrix = m_selectionContext.getComponent<TransformComponent>().
                                    getTransform();
                            glm::mat4 worldMatrix = parentMatrix * transform.getTransform();
                            transform.setTransform(worldMatrix);

                            // Setup MeshComponent with quadric parameters.
                            auto &mesh = entityInstance.getOrAddComponent<MeshComponent>(QUADRIC);
                            mesh.polygonMode() = VK_POLYGON_MODE_LINE;
                            auto quadricParams = std::dynamic_pointer_cast<QuadricMeshParameters>(mesh.meshParameters);

                            // Setup MaterialComponent.
                            auto &material = entityInstance.getOrAddComponent<MaterialComponent>();
                            material.useTexture = true;

                            // Downsampled data.
                            quadricParams->a = comp.a[i];
                            quadricParams->b = comp.b[i];
                            quadricParams->c = comp.c[i];
                            quadricParams->t_x = comp.t_x[i];
                            quadricParams->t_y = comp.t_y[i];
                            quadricParams->kernelScale = comp.kernelScale[i];
                            quadricParams->threshold = comp.threshold[i];
                            quadricParams->b_beta = comp.beta[i];
                        }
                    }
                }
                break;
                case LayerUtils::PLY_MESH:
                    if (m_selectionContext.hasComponent<MeshComponent>()) {
                        auto &meshComponent = m_selectionContext.getComponent<MeshComponent>();
                        meshComponent.meshParameters = std::make_shared<PLYFileMeshParameters>(loadFileInfo.path);
                    }
                    break;
                default:
                    Log::Logger::getInstance()->warning("Not implemented yet");
                    break;
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
        if (m_loadFileFuture.valid() &&
            m_loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            LayerUtils::LoadFileInfo loadFileInfo = m_loadFileFuture.get(); // Get the result from the future
            handleSelectedFileOrFolder(loadFileInfo);
        }
    }

    void PropertiesLayer::checkFolderImportCompletion() {
        if (m_loadFolderFuture.valid() &&
            m_loadFolderFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            LayerUtils::LoadFileInfo loadFileInfo = m_loadFolderFuture.get(); // Get the result from the future
            handleSelectedFileOrFolder(loadFileInfo);
        }
    }
}
