//
// Created by magnus on 8/23/24.
//

#include "Viewer/VkRender/Editors/Common/Properties/PropertiesLayer.h"

#include <Viewer/VkRender/Components/GaussianComponent.h>

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/PointCloudComponent.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Editors/Common/CommonEditorFunctions.h"

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

    bool PropertiesLayer::drawVec3Control(const std::string& label, glm::vec3& values, float resetValue = 0.0f,
                                          float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO& io = ImGui::GetIO();
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

    bool PropertiesLayer::drawFloatControl(const std::string& label, float& value, float resetValue = 0.0f,
                                           float speed = 1.0f, float columnWidth = 100.0f) {
        bool valueChanged = false;
        ImGuiIO& io = ImGui::GetIO();
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
        if (ImGui::Button("X", buttonSize)) {
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

    template <typename T, typename UIFunction>
    void PropertiesLayer::drawComponent(const std::string& componentName, Entity entity, UIFunction uiFunction) {
        const ImGuiTreeNodeFlags treeNodeFlags =
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth |
            ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap;
        if (entity.hasComponent<T>()) {
            auto& component = entity.getComponent<T>();
            ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
            float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
            ImGui::Separator();
            bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, "%s", componentName.c_str());
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
            displayAddComponentEntry<CameraComponent>("Camera");
            displayAddComponentEntry<TransformComponent>("Transform");
            displayAddComponentEntry<MeshComponent>("Mesh");
            displayAddComponentEntry<MaterialComponent>("Material");
            displayAddComponentEntry<PointCloudComponent>("PointCloud");
            displayAddComponentEntry<GaussianComponent>("3DGS Model");

            ImGui::EndPopup();
        }

        drawComponent<TransformComponent>("Transform", entity, [](auto& component) {
            drawVec3Control("Translation", component.getPosition());
            drawVec3Control("Rotation", component.getRotationEuler(), 0.0f, 2.0f);
            component.updateFromEulerRotation();
            drawVec3Control("Scale", component.getScale(), 1.0f);
        });
        drawComponent<CameraComponent>("Camera", entity, [](auto& component) {
            drawFloatControl("Field of View", component.camera->fov(), 1.0f);
            component.camera->updateProjectionMatrix();

            ImGui::Checkbox("Render scene from viewpoint", &component.renderFromViewpoint());

            if (component.renderFromViewpoint())
                component.camera->setType(Camera::CameraType::flycam);
        });

        drawComponent<MeshComponent>("Mesh", entity, [this, entity](auto& component) {
            ImGui::Text("MeshFile:");
            ImGui::Text("%s", component.m_meshPath.string().c_str());


            // Load mesh from file here:
            if (ImGui::Button("Load .obj mesh")) {
                std::vector<std::string> types{".obj"};
                EditorUtils::openImportFileDialog("Wavefront", types, LayerUtils::OBJ_FILE, &m_loadFileFuture);
            }

            ImGui::SameLine();
            // Load mesh from mesh here:
            if (ImGui::Button("Load .ply mesh")) {
                std::vector<std::string> types{".ply"};
                EditorUtils::openImportFileDialog("Stanford .PLY", types, LayerUtils::PLY_MESH, &m_loadFileFuture);
            }

            // Polygon Mode Control
            ImGui::Text("Polygon Mode:");
            const char* polygonModes[] = {"Line", "Fill"};
            int currentMode = (component.polygonMode == VK_POLYGON_MODE_LINE) ? 0 : 1;
            if (ImGui::Combo("Polygon Mode", &currentMode, polygonModes, IM_ARRAYSIZE(polygonModes))) {
                if (currentMode == 0) {
                    component.polygonMode = VK_POLYGON_MODE_LINE;
                }
                else {
                    component.polygonMode = VK_POLYGON_MODE_FILL;
                }
                // notify component updated
                // m_scene.onComponentUpdated(entity, component);
                m_context->activeScene()->onComponentUpdated(entity, component);
            }
        });
        drawComponent<TagComponent>("Tag", entity, [this](auto& component) {
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

        drawComponent<MaterialComponent>("Material", entity, [this, entity](auto& component) {
            ImGui::Text("Material Properties");

            // Base Color Control
            ImGui::Text("Base Color");
            ImGui::ColorEdit4("##BaseColor", glm::value_ptr(component.baseColor));
            /*
            // Metallic Control
            ImGui::Text("Metallic");
            ImGui::SliderFloat("##Metallic", &component.metallic, 0.0f, 1.0f);

            // Roughness Control
            ImGui::Text("Roughness");
            ImGui::SliderFloat("##Roughness", &component.roughness, 0.0f, 1.0f);


            // Emissive Factor Control
            ImGui::Text("Emissive Factor");
            ImGui::ColorEdit3("##EmissiveFactor", glm::value_ptr(component.emissiveFactor));
*/

            if (ImGui::Button("Reload Material Shader")) {
                component.reloadShader = true;
                m_context->activeScene()->onComponentUpdated(entity, component);
            }
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

            // Texture Control
            ImGui::Checkbox("Use Video Source", &component.usesVideoSource);
            if (component.usesVideoSource) {
                ImGui::BeginChild("TextureChildWindow", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 6), true);

                ImGui::Text("Images folder:");
                ImGui::Text("%s", component.videoFolderSource.string().c_str());
                // Button to load texture
                if (ImGui::Button("Set Image Folder")) {
                    std::vector<std::string> types{".png", ".jpg", ".bmp"};
                    EditorUtils::openImportFolderDialog("Set Image Folder", types, LayerUtils::VIDEO_TEXTURE_FILE,
                                                        &m_loadFolderFuture);
                }

                ImGui::Checkbox("Is Disparity", &component.isDisparity);


                ImGui::EndChild();
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

        drawComponent<PointCloudComponent>("PointCloud", entity, [this](auto& component) {
            ImGui::Text("Point Size");
            ImGui::SliderFloat("##PointSize", &component.pointSize, 0.0f, 10.0f);
            // Texture Control
            ImGui::Checkbox("Use Video Source", &component.usesVideoSource);
            if (component.usesVideoSource) {
                ImGui::BeginChild("TextureChildWindow", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 6), true);
                ImGui::Text("Depth Images folder:");
                ImGui::Text("%s", component.depthVideoFolderSource.string().c_str());
                // Button to load texture
                if (ImGui::Button("Set Depth Images")) {
                    std::vector<std::string> types{".png", ".jpg", ".bmp"};
                    EditorUtils::openImportFolderDialog("Set Image Folder", types,
                                                        LayerUtils::VIDEO_DISPARITY_DEPTH_TEXTURE_FILE,
                                                        &m_loadFolderFuture);
                }
                ImGui::Text("Color Images folder:");
                ImGui::Text("%s", component.colorVideoFolderSource.string().c_str());
                // Button to load texture
                if (ImGui::Button("Set Color Images")) {
                    std::vector<std::string> types{".png", ".jpg", ".bmp"};
                    EditorUtils::openImportFolderDialog("Set Image Folder", types,
                                                        LayerUtils::VIDEO_DISPARITY_COLOR_TEXTURE_FILE,
                                                        &m_loadFolderFuture);
                }
                ImGui::EndChild();
            }
        });
        drawComponent<GaussianComponent>("Gaussian Model", entity, [this](auto& component) {
            ImGui::Text("Gaussian Model Properties");

            // Display the number of Gaussians
            ImGui::Text("Number of Gaussians: %zu", component.size());

            ImGui::Separator();

            // Button to add a new Gaussian
            component.addToRenderer = ImGui::Button("Add Gaussian");
            if (component.addToRenderer) {
                // Default values for a new Gaussian
                glm::vec3 defaultMean(0.0f, 0.0f, 0.0f);
                glm::vec3 defaultScale(0.3f); // Identity matrix
                glm::quat defaultQuat(1.0f, 0.0f, 0.0f, 0.0f); // Identity matrix
                float defaultOpacity = 1.0f;
                glm::vec3 color(1.0f, 0.0f, 0.0f);
                component.addGaussian(defaultMean, defaultScale, defaultQuat, defaultOpacity, color);
            }
            ImGui::SameLine();
            if(ImGui::Button("Load from file")){
                std::vector<std::string> types{".ply"};
                EditorUtils::openImportFileDialog("Load 3DGS .ply file", types, LayerUtils::PLY_3DGS,
                                                  &m_loadFileFuture);
            }


            ImGui::Spacing();

            // Iterate over each Gaussian and provide controls to modify them
            if (component.size() < 10) {
                for (size_t i = 0; i < component.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i)); // Ensure unique ID for ImGui widgets
                    // Collapsible header for each Gaussian
                    if (ImGui::CollapsingHeader(("Gaussian " + std::to_string(i)).c_str())) {
                        // Mean Position Controls
                        component.addToRenderer |= drawVec3Control("Position", component.means[i], 0.0f);
                        component.addToRenderer |= drawVec3Control("Scale", component.scales[i], 0.0f, 0.1f);
                        component.addToRenderer |= drawVec3Control("Color", component.colors[i], 0.0f, 0.1f);
                        // Amplitude Control
                        ImGui::Text("Opacity");

                        component.addToRenderer |= ImGui::DragFloat("##Opacity", &component.opacities[i], 0.1f, 0.0f,
                                                                    10.0f);


                        // Button to remove this Gaussian
                        ImGui::Spacing();
                        if (ImGui::Button("Remove Gaussian")) {
                            component.means.erase(component.means.begin() + i);
                            component.scales.erase(component.scales.begin() + i);
                            component.opacities.erase(component.opacities.begin() + i);
                            component.rotations.erase(component.rotations.begin() + i);
                            --i; // Adjust index after removal
                        }
                    }

                    ImGui::PopID(); // Pop ID for this Gaussian
                }
            }
        });
    }


    /** Called once per frame **/
    void PropertiesLayer::onUIRender() {
        m_selectionContext = m_context->getSelectedEntity();
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
        checkFolderImportCompletion();

        ImGui::End();
    }

    /** Called once upon this object destruction **/
    void PropertiesLayer::onDetach() {
    }

    template <typename T>
    void PropertiesLayer::displayAddComponentEntry(const std::string& entryName) {
        if (!m_selectionContext.hasComponent<T>()) {
            if (ImGui::MenuItem(entryName.c_str())) {
                m_selectionContext.addComponent<T>();
                // Check if the component added is a PointCloudComponent
                if constexpr (std::is_same_v<T, PointCloudComponent>) {
                    if (!m_selectionContext.hasComponent<MeshComponent>()) {
                        auto& component = m_selectionContext.addComponent<MeshComponent>();
                        component.m_type = MeshDataType::POINT_CLOUD;
                        component.polygonMode = VK_POLYGON_MODE_POINT;
                    }
                }
                ImGui::CloseCurrentPopup();
            }
        }
    }

    void
    PropertiesLayer::handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo& loadFileInfo) {
        if (!loadFileInfo.path.empty()) {
            switch (loadFileInfo.filetype) {
            case LayerUtils::TEXTURE_FILE:
                {
                    auto& materialComponent = m_selectionContext.getComponent<MaterialComponent>();
                    materialComponent.albedoTexturePath = loadFileInfo.path;
                    m_context->activeScene()->onComponentUpdated(m_selectionContext, materialComponent);
                }
                break;
            case LayerUtils::OBJ_FILE:
                // Load into the active scene
                if (m_selectionContext.hasComponent<MeshComponent>())
                    m_selectionContext.removeComponent<MeshComponent>();
                m_selectionContext.addComponent<MeshComponent>(loadFileInfo.path, MeshDataType::OBJ_FILE);
                break;
            case LayerUtils::PLY_3DGS:
                {

                    if (m_selectionContext.hasComponent<GaussianComponent>())
                        m_selectionContext.removeComponent<GaussianComponent>();

                    auto& comp = m_selectionContext.addComponent<GaussianComponent>(loadFileInfo.path);
                    comp.addToRenderer = true;
                }
                break;
            case LayerUtils::PLY_MESH:
                {
                    if (m_selectionContext.hasComponent<MeshComponent>())
                        m_selectionContext.removeComponent<MeshComponent>();
                    m_selectionContext.addComponent<MeshComponent>(loadFileInfo.path, MeshDataType::PLY_FILE);
                }
                break;
            case LayerUtils::VIDEO_TEXTURE_FILE:
                {
                    // TODO figure out how to know which component requested the folder or file load operation
                    if (m_selectionContext.hasComponent<MaterialComponent>()) {
                        auto& materialComponent = m_selectionContext.getComponent<MaterialComponent>();
                        materialComponent.videoFolderSource = loadFileInfo.path;
                        m_context->activeScene()->onComponentUpdated(m_selectionContext, materialComponent);
                    }
                }
                break;
            case LayerUtils::VIDEO_DISPARITY_DEPTH_TEXTURE_FILE:
                if (m_selectionContext.hasComponent<PointCloudComponent>()) {
                    auto& pointCloudComponent = m_selectionContext.getComponent<PointCloudComponent>();
                    pointCloudComponent.depthVideoFolderSource = loadFileInfo.path;
                    m_context->activeScene()->onComponentUpdated(m_selectionContext, pointCloudComponent);
                }
                break;
            case LayerUtils::VIDEO_DISPARITY_COLOR_TEXTURE_FILE:
                if (m_selectionContext.hasComponent<PointCloudComponent>()) {
                    auto& pointCloudComponent = m_selectionContext.getComponent<PointCloudComponent>();
                    pointCloudComponent.colorVideoFolderSource = loadFileInfo.path;
                    m_context->activeScene()->onComponentUpdated(m_selectionContext, pointCloudComponent);
                }
                break;
            default:
                Log::Logger::getInstance()->warning("Not implemented yet");
                break;
            }

            // Copy the selected file path to wherever it's needed
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            opts.lastOpenedImportModelFolderPath = loadFileInfo.path;
            // Additional processing of the file can be done here
            Log::Logger::getInstance()->info("File selected: {}", loadFileInfo.path.filename().string());
        }
        else {
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
