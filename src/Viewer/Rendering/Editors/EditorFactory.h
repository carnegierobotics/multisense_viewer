//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORFACTORY_H
#define MULTISENSE_VIEWER_EDITORFACTORY_H

#include <utility>

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Editors/Common/3DViewport/Editor3DViewport.h"
#include "Viewer/Rendering/Editors/Common/SceneHierarchy/EditorSceneHierarchy.h"
#include "Viewer/Rendering/Editors/MultiSenseViewer/SidebarEditor/SideBarEditor.h"
#include "Viewer/Rendering/Editors/MultiSenseViewer/ConfigurationEditor/ConfigurationEditor.h"
#include "Viewer/Rendering/Editors/Common/Test/EditorTest.h"
#include "Viewer/Rendering/Editors/Common/Properties/EditorProperties.h"
#include "Viewer/Rendering/Editors/Common/ImageEditor/EditorImage.h"
#include "Viewer/Rendering/Editors/Common/SceneRenderer.h"
#ifdef SYCL_ENABLED
#include "Viewer/Rendering/Editors/Common/GaussianViewer/EditorGaussianViewer.h"
#endif
namespace VkRender {

    // Define the factory class
    class EditorFactory {
    public:
        EditorFactory(){
            registerEditor(EditorType::SceneRenderer, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<SceneRenderer>(ci, uuid);
            });
            registerEditor(EditorType::SceneHierarchy, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorSceneHierarchy>(ci, uuid);
            });
            registerEditor(EditorType::MultiSenseViewer_Sidebar, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<SideBarEditor>(ci, uuid);
            });
            registerEditor(EditorType::MultiSenseViewer_Configuration, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<ConfigurationEditor>(ci, uuid);
            });
            registerEditor(EditorType::Viewport3D, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<VkRender::Editor3DViewport>(ci, uuid);
            });
            registerEditor(EditorType::ImageEditor, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorImage>(ci, uuid);
            });
            registerEditor(EditorType::TestWindow, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorTest>(ci, uuid);
            });
            registerEditor(EditorType::Properties, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorProperties>(ci, uuid);
            });
#ifdef SYCL_ENABLED
            registerEditor(EditorType::GaussianViewer, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorGaussianViewer>(ci, uuid);
            });
#endif
        }

        using CreatorFunc = std::function<std::unique_ptr<Editor>(EditorCreateInfo&, UUID uuid)>;

        void registerEditor(EditorType type, CreatorFunc func) {
            m_creators[type] = std::move(func);
        }

        std::unique_ptr<Editor> createEditor(EditorType type, EditorCreateInfo &createInfo, UUID uuid) {
            auto it = m_creators.find(type);
            if (it != m_creators.end()) {
                return it->second(createInfo, uuid);
            }
            Log::Logger::getInstance()->warning("Failed to find editorType: {} in factory, reverting to {}", editorTypeToString(type),
                                                editorTypeToString(EditorType::TestWindow));
            createInfo.editorTypeDescription = EditorType::TestWindow;
            return m_creators[EditorType::TestWindow](createInfo, uuid);
        }

    private:
        std::unordered_map<EditorType, CreatorFunc> m_creators;
    };

}

#endif //MULTISENSE_VIEWER_EDITORFACTORY_H
