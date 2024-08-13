//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORFACTORY_H
#define MULTISENSE_VIEWER_EDITORFACTORY_H

#include <utility>

#include "Viewer/VkRender/pch.h"

#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Editors/Common/Viewport/EditorViewport.h"
#include "Viewer/VkRender/Editors/Common/SceneHierarchy/EditorSceneHierarchy.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/EditorMultiSenseViewer.h"
#include "Viewer/VkRender/Editors/Common/Test/EditorTest.h"
#include "Viewer/VkRender/Editors/MyEditor/EditorMyProject.h"

namespace VkRender {



    // Define the factory class
    class EditorFactory {
    public:
        explicit EditorFactory(const VulkanRenderPassCreateInfo& createInfo)
                : m_defaultCreateInfo(const_cast<VulkanRenderPassCreateInfo&>(createInfo)) {
            registerEditor(EditorType::SceneHierarchy, [](VulkanRenderPassCreateInfo &ci, UUID) {
                return std::make_unique<EditorSceneHierarchy>(ci);
            });
            registerEditor(EditorType::MultiSenseViewer, [](VulkanRenderPassCreateInfo &ci, UUID) {
                return std::make_unique<EditorMultiSenseViewer>(ci);
            });
            registerEditor(EditorType::Viewport, [](VulkanRenderPassCreateInfo &ci, UUID) {
                return std::make_unique<EditorViewport>(ci);
            });
            registerEditor(EditorType::MyProject, [](VulkanRenderPassCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorMyProject>(ci, uuid);
            });
            registerEditor(EditorType::TestWindow, [](VulkanRenderPassCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorTest>(ci, uuid);
            });
        }

        using CreatorFunc = std::function<std::unique_ptr<Editor>(VulkanRenderPassCreateInfo&, UUID)>;

        void registerEditor(EditorType type, CreatorFunc func) {
            m_creators[type] = std::move(func);
        }

        std::unique_ptr<Editor> createEditor(EditorType type, VulkanRenderPassCreateInfo &createInfo, UUID uuid) {
            auto it = m_creators.find(type);
            if (it != m_creators.end()) {
                return it->second(createInfo, uuid);
            }
            Log::Logger::getInstance()->warning("Failed to find editorType: {} in factory, reverting to {}", editorTypeToString(type),
                                                editorTypeToString(EditorType::TestWindow));

            return std::make_unique<Editor>(m_defaultCreateInfo);
        }

    private:
        std::unordered_map<EditorType, CreatorFunc> m_creators;
        VulkanRenderPassCreateInfo& m_defaultCreateInfo;
    };

}

#endif //MULTISENSE_VIEWER_EDITORFACTORY_H
