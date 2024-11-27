//
// Created by magnus on 8/23/24.
//

#include "PropertiesLayer.h"

#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"

#include "Viewer/Rendering/ImGui/Layer.h"

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Application/Application.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

namespace VkRender {

    struct CameraModel {
        enum Type {
            PINHOLE,
            SIMPLE_PINHOLE,
            OPENCV,
            SIMPLE_RADIAL,
            // Add other models as needed
            UNKNOWN
        };

        static Type fromString(const std::string &modelStr) {
            if (modelStr == "PINHOLE") return PINHOLE;
            if (modelStr == "SIMPLE_PINHOLE") return SIMPLE_PINHOLE;
            if (modelStr == "OPENCV") return OPENCV;
            if (modelStr == "SIMPLE_RADIAL") return SIMPLE_RADIAL;
            return UNKNOWN;
        }
    };

    struct ColmapCamera {
        uint32_t camera_id;
        CameraModel::Type model;
        uint32_t width;
        uint32_t height;
        std::vector<double> params; // Intrinsic parameters
    };

    struct Image {
        uint32_t image_id;
        glm::quat rotation; // Quaternion representing rotation (qw, qx, qy, qz)
        glm::vec3 translation; // Translation vector (tx, ty, tz)
        uint32_t camera_id;
        std::string image_name;
        // Add other fields if needed
    };

    bool
    parseCameras(const std::filesystem::path &camerasFilePath, std::unordered_map<uint32_t, ColmapCamera> &cameras) {
        std::ifstream file(camerasFilePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << camerasFilePath << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            uint32_t camera_id;
            std::string model_str;
            uint32_t width, height;
            std::vector<double> params;

            iss >> camera_id >> model_str >> width >> height;

            CameraModel::Type model = CameraModel::fromString(model_str);
            if (model == CameraModel::UNKNOWN) {
                std::cerr << "Unknown camera model: " << model_str << std::endl;
                continue;
            }

            double param;
            while (iss >> param) {
                params.push_back(param);
            }

            ColmapCamera camera = {camera_id, model, width, height, params};
            cameras[camera_id] = camera;
        }

        return true;
    }

    bool parseImages(const std::filesystem::path &imagesFilePath, std::unordered_map<uint32_t, Image> &images) {
        std::ifstream file(imagesFilePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << imagesFilePath << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            uint32_t image_id;
            double qw, qx, qy, qz;
            double tx, ty, tz;
            uint32_t camera_id;
            std::string image_name;

            iss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> image_name;
            // Handle image names with spaces
            while (iss >> line) {
                image_name += " " + line;
            }

            // Read and discard the next line (2D points)
            std::getline(file, line);

            Image image;
            image.image_id = image_id;
            image.rotation = glm::quat(qw, qx, qy, qz);
            image.translation = glm::vec3(tx, ty, tz);
            image.camera_id = camera_id;
            image.image_name = image_name;

            images[image_id] = image;
        }

        return true;
    }

    struct FOV {
        double horizontal; // In radians
        double vertical;   // In radians
    };

    FOV computeFOV(const ColmapCamera &camera) {
        FOV fov = {0.0, 0.0};

        if (camera.model == CameraModel::PINHOLE || camera.model == CameraModel::OPENCV) {
            // For PINHOLE and OPENCV models, params are [fx, fy, cx, cy]
            double fx = camera.params[0];
            double fy = camera.params[1];
            fov.horizontal = 2.0 * atan(camera.width / (2.0 * fx));
            fov.vertical = 2.0 * atan(camera.height / (2.0 * fy));
        } else if (camera.model == CameraModel::SIMPLE_PINHOLE) {
            // For SIMPLE_PINHOLE, params are [f, cx, cy]
            double f = camera.params[0];
            fov.horizontal = 2.0 * atan(camera.width / (2.0 * f));
            fov.vertical = 2.0 * atan(camera.height / (2.0 * f));
        } else {
            std::cerr << "FOV computation not implemented for this camera model." << std::endl;
        }

        return fov;
    }

    void PropertiesLayer::addEntitiesFromColmap(const std::filesystem::path &colmapFolderPath) {
        std::filesystem::path imagesTXTfile = colmapFolderPath / "sparse" / "0" / "images.txt";
        std::filesystem::path camerasTXTfile = colmapFolderPath / "sparse" / "0" / "cameras.txt";

        std::unordered_map<uint32_t, ColmapCamera> cameras;
        if (!parseCameras(camerasTXTfile, cameras)) {
            std::cerr << "Failed to parse cameras.txt" << std::endl;
        }

        std::unordered_map<uint32_t, Image> images;
        if (!parseImages(imagesTXTfile, images)) {
            std::cerr << "Failed to parse images.txt" << std::endl;
        }

        // Now, for each image, get the camera info and compute FOV
        for (const auto &[image_id, image]: images) {
            auto camera_it = cameras.find(image.camera_id);
            if (camera_it == cameras.end()) {
                std::cerr << "Camera ID " << image.camera_id << " not found for image " << image.image_name
                          << std::endl;
                continue;
            }

            const ColmapCamera &camera = camera_it->second;

            // Compute FOV
            FOV fov = computeFOV(camera);

            // Output the information
            std::cout << "Image ID: " << image.image_id << std::endl;
            std::cout << "Image Name: " << image.image_name << std::endl;
            std::cout << "Camera ID: " << image.camera_id << std::endl;
            std::cout << "Position (T): [" << image.translation.x << ", " << image.translation.y << ", "
                      << image.translation.z << "]" << std::endl;
            std::cout << "Rotation (Q): [" << image.rotation.w << ", " << image.rotation.x << ", " << image.rotation.y
                      << ", " << image.rotation.z << "]" << std::endl;
            std::cout << "Width: " << camera.width << ", Height: " << camera.height << std::endl;
            std::cout << "FOV Horizontal (degrees): " << glm::degrees(fov.horizontal) << std::endl;
            std::cout << "FOV Vertical (degrees): " << glm::degrees(fov.vertical) << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            // You can now create entities or perform further processing with this data
            auto scene = m_context->activeScene();
            auto entity = scene->createEntity(image.image_name);
            Camera newCamera(camera.width, camera.height);
            newCamera.setType(Camera::flycam);
            newCamera.fov() = std::min(glm::degrees(fov.horizontal), glm::degrees(fov.vertical));

            auto &cameraComponent = entity.addComponent<CameraComponent>(newCamera);
            cameraComponent.renderFromViewpoint() = false;

            auto &transformComponent = entity.getComponent<TransformComponent>();
            glm::quat q_wc(image.rotation.w, image.rotation.x, image.rotation.y, image.rotation.z);
            glm::quat q_cw = glm::conjugate(q_wc);
            glm::vec3 position = -(q_cw * image.translation);


            glm::quat q(1.0f, 0.0f, 0.0f, 0.0f);
            transformComponent.setPosition(position);
            transformComponent.setRotationQuaternion(q_cw * glm::quat(0.0f, 1.0f, 0.0f, 0.0f));
            //transformComponent.setRotationQuaternion(q_cw);
            transformComponent.setScale({0.5f, 0.5f, 0.5f});

            auto &material = entity.addComponent<MaterialComponent>();
            material.fragmentShaderName = "defaultTexture.frag";
            material.albedoTexturePath = "";


            entity.setParent(m_selectionContext);
            m_selectionContext.getComponent<GroupComponent>().colmapPath = colmapFolderPath;
        }
    }

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

    template<typename T, typename UIFunction>
    void PropertiesLayer::drawComponent(const std::string &componentName, Entity entity, UIFunction uiFunction) {
        const ImGuiTreeNodeFlags treeNodeFlags =
                ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth |
                ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_AllowOverlap;
        if (entity.hasComponent<T>()) {
            auto &component = entity.getComponent<T>();
            ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
            float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
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
            displayAddComponentEntry<CameraComponent>("Camera");
            displayAddComponentEntry<TransformComponent>("Transform");
            displayAddComponentEntry<MeshComponent>("Mesh");
            displayAddComponentEntry<MaterialComponent>("Material");
            displayAddComponentEntry<PointCloudComponent>("PointCloud");
            displayAddComponentEntry<GaussianComponent>("3DGS Model");
            displayAddComponentEntry<VectorComponent>("VectorComponent");

            ImGui::EndPopup();
        }

        drawComponent<TransformComponent>("Transform", entity, [](auto &component) {
            drawVec3Control("Translation", component.getPosition());
            drawVec3Control("Rotation", component.getRotationEuler(), 0.0f, 2.0f);
            component.updateFromEulerRotation();
            drawVec3Control("Scale", component.getScale(), 1.0f);
        });
        drawComponent<CameraComponent>("Camera", entity, [](auto &component) {
            drawFloatControl("Field of View", component.camera->fov(), 1.0f);
            component.camera->updateProjectionMatrix();

            ImGui::Checkbox("Render scene from viewpoint", &component.renderFromViewpoint());
            // Controls for Camera Resolution (Width and Height)
            int width = static_cast<int>(component.camera->width());
            int height = static_cast<int>(component.camera->height());
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputInt("Width", &width)) {
                if (width > 0) {
                    component.camera->setCameraResolution(static_cast<uint32_t>(width), component.camera->height());
                    component.camera->updateProjectionMatrix();
                }
            }
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::InputInt("Height", &height)) {
                if (height > 0) {
                    component.camera->setCameraResolution(component.camera->width(), static_cast<uint32_t>(height));
                    component.camera->updateProjectionMatrix();
                }
            }

            if (component.renderFromViewpoint())
                component.camera->setType(Camera::CameraType::flycam);
        });

        drawComponent<MeshComponent>("Mesh", entity, [this, entity](auto &component) {
            ImGui::Text("MeshFile:");
            ImGui::Text("%s", component.modelPath().string().c_str());


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

            ImGui::SameLine();
            // Load mesh from mesh here:
            if (ImGui::Button("Camera gizmo")) {
                component.meshDataType() = CAMERA_GIZMO;
                if (m_selectionContext.hasComponent<MeshComponent>())
                    m_selectionContext.removeComponent<MeshComponent>();
                m_selectionContext.addComponent<MeshComponent>(MeshDataType::CAMERA_GIZMO);
            }

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

        drawComponent<MaterialComponent>("Material", entity, [this, entity](auto &component) {
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

        /*
        drawComponent<PointCloudComponent>("PointCloud", entity, [this](auto &component) {
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
        */
        drawComponent<GaussianComponent>("Gaussian Model", entity, [this](auto &component) {
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
            if (ImGui::Button("Load from file")) {
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


        drawComponent<GroupComponent>("Group", entity, [this](auto &component) {
            ImGui::Text("Load cameras from file");
            ImGui::Text("Colmap Path: %s", component.colmapPath.string().c_str());
            if (ImGui::Button("Set Colmap Folder")) {
                std::vector<std::string> types{""};
                EditorUtils::openImportFolderDialog("Load 3DGS .ply file", types, LayerUtils::COLMAP_FOLDER,
                                                    &m_loadFileFuture);

            }
            ImGui::SameLine();
            if (ImGui::Button("Load colmap from path")) {
                if (std::filesystem::exists(component.colmapPath))
                    addEntitiesFromColmap(component.colmapPath);
            }
        });

        drawComponent<VectorComponent>("VectorComponent", entity, [this](auto &component) {
            ImGui::Text("Vector:");

            bool update = false;

            update |= drawVec3Control("Origin", component.origin);
            update |= drawVec3Control("Direction", component.direction);
            update |= ImGui::SliderFloat("Magnitude", &component.magnitude, 0.0f, 100.0f);

            if (update){
                // Normalize the direction to ensure consistent scaling
                component.direction = glm::normalize(component.direction);

                // Calculate the endpoint of the vector
                glm::vec3 endpoint = component.origin + component.direction * component.magnitude;

                // Cylinder parameters
                const int segments = 16; // Number of segments for the circular caps
                float radius = 0.05f * component.magnitude; // Cylinder radius

                // Base and top centers
                glm::vec3 baseCenter = component.origin;
                glm::vec3 topCenter = endpoint;

                // Compute orthogonal vectors for the circular base
                glm::vec3 up = glm::vec3(0, 1, 0); // Arbitrary perpendicular vector
                if (glm::abs(glm::dot(up, component.direction)) > 0.9f) {
                    up = glm::vec3(1, 0, 0); // Switch to another perpendicular vector if needed
                }
                glm::vec3 right = glm::normalize(glm::cross(component.direction, up));
                up = glm::normalize(glm::cross(right, component.direction));

                std::vector<Vertex> vertices;
                std::vector<uint32_t> indices;

                // Generate vertices for the bottom and top circles
                for (int i = 0; i < segments; ++i) {
                    float theta = (2.0f * M_PI * i) / segments; // Angle around the circle

                    // Circular offset
                    glm::vec3 offset = right * cos(theta) * radius + up * sin(theta) * radius;

                    // Bottom circle vertex
                    vertices.push_back({ baseCenter + offset });

                    // Top circle vertex
                    vertices.push_back({ topCenter + offset });
                }

                // Add center vertices for the caps
                uint32_t baseCenterIndex = static_cast<uint32_t>(vertices.size());
                vertices.push_back({ baseCenter }); // Bottom center
                uint32_t topCenterIndex = static_cast<uint32_t>(vertices.size());
                vertices.push_back({ topCenter });  // Top center

                // Generate indices for the side faces
                for (int i = 0; i < segments; ++i) {
                    uint32_t next = (i + 1) % segments;

                    // First triangle of quad
                    indices.push_back(i);               // Bottom circle vertex
                    indices.push_back(segments + i);    // Top circle vertex
                    indices.push_back(segments + next); // Next top circle vertex

                    // Second triangle of quad
                    indices.push_back(i);           // Bottom circle vertex
                    indices.push_back(segments + next); // Next top circle vertex
                    indices.push_back(next);        // Next bottom circle vertex
                }

                // Generate indices for the bottom cap
                for (int i = 0; i < segments; ++i) {
                    uint32_t next = (i + 1) % segments;

                    indices.push_back(baseCenterIndex); // Bottom center
                    indices.push_back(i);               // Current bottom circle vertex
                    indices.push_back(next);            // Next bottom circle vertex
                }

                // Generate indices for the top cap
                for (int i = 0; i < segments; ++i) {
                    uint32_t next = (i + 1) % segments;

                    indices.push_back(topCenterIndex);  // Top center
                    indices.push_back(segments + next); // Next top circle vertex
                    indices.push_back(segments + i);    // Current top circle vertex
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
                    if (m_selectionContext.hasComponent<MeshComponent>())
                        m_selectionContext.removeComponent<MeshComponent>();
                    m_selectionContext.addComponent<MeshComponent>(MeshDataType::OBJ_FILE, loadFileInfo.path);
                    break;
                case LayerUtils::PLY_3DGS: {

                    if (m_selectionContext.hasComponent<GaussianComponent>())
                        m_selectionContext.removeComponent<GaussianComponent>();

                    auto &comp = m_selectionContext.addComponent<GaussianComponent>(loadFileInfo.path);
                    comp.addToRenderer = true;
                }
                    break;
                case LayerUtils::PLY_MESH: {
                    if (m_selectionContext.hasComponent<MeshComponent>())
                        m_selectionContext.removeComponent<MeshComponent>();
                    m_selectionContext.addComponent<MeshComponent>(MeshDataType::PLY_FILE, loadFileInfo.path);
                }
                    break;
                case LayerUtils::VIDEO_TEXTURE_FILE: {
                    // TODO figure out how to know which component requested the folder or file load operation
                    if (m_selectionContext.hasComponent<MaterialComponent>()) {
                        auto &materialComponent = m_selectionContext.getComponent<MaterialComponent>();
                        materialComponent.videoFolderSource = loadFileInfo.path;
                        m_context->activeScene()->onComponentUpdated(m_selectionContext, materialComponent);
                    }
                }
                    break;
                case LayerUtils::VIDEO_DISPARITY_DEPTH_TEXTURE_FILE:
                    if (m_selectionContext.hasComponent<PointCloudComponent>()) {
                        auto &pointCloudComponent = m_selectionContext.getComponent<PointCloudComponent>();
                        pointCloudComponent.depthVideoFolderSource = loadFileInfo.path;
                        m_context->activeScene()->onComponentUpdated(m_selectionContext, pointCloudComponent);
                    }
                    break;
                case LayerUtils::VIDEO_DISPARITY_COLOR_TEXTURE_FILE:
                    if (m_selectionContext.hasComponent<PointCloudComponent>()) {
                        auto &pointCloudComponent = m_selectionContext.getComponent<PointCloudComponent>();
                        pointCloudComponent.colorVideoFolderSource = loadFileInfo.path;
                        m_context->activeScene()->onComponentUpdated(m_selectionContext, pointCloudComponent);
                    }
                    break;
                case LayerUtils::COLMAP_FOLDER:
                    if (m_selectionContext.hasComponent<GroupComponent>()) {
                        auto &groupComponent = m_selectionContext.getComponent<GroupComponent>();
                        std::filesystem::path colmapFolderPath = loadFileInfo.path;
                        addEntitiesFromColmap(colmapFolderPath);
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
