//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORFACTORY_H
#define MULTISENSE_VIEWER_EDITORFACTORY_H

#include <utility>

#include "Viewer/VkRender/pch.h"

#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Editors/Viewport/EditorViewport.h"
#include "Viewer/VkRender/Editors/SceneHierarchy/EditorSceneHierarchy.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/EditorMultiSenseViewer.h"
#include "Viewer/VkRender/Editors/Test/EditorTest.h"

namespace VkRender {



    // Define the factory class
    class EditorFactory {
    public:

        explicit EditorFactory(const VulkanRenderPassCreateInfo& createInfo) : m_defaultCreateInfo(
                const_cast<VulkanRenderPassCreateInfo &>(createInfo)) {
            registerEditor(EditorType::SceneHierarchy, [](VulkanRenderPassCreateInfo &ci, UUID) { return EditorSceneHierarchy(ci); });
            registerEditor(EditorType::MultiSenseViewer, [](VulkanRenderPassCreateInfo &ci, UUID) { return EditorMultiSenseViewer(ci); });
            registerEditor(EditorType::Viewport, [](VulkanRenderPassCreateInfo &ci, UUID) { return EditorViewport(ci); });
            registerEditor(EditorType::TestWindow, [](VulkanRenderPassCreateInfo &ci, UUID uuid) { return EditorTest(ci, uuid); });
        }
        using CreatorFunc = std::function<Editor(VulkanRenderPassCreateInfo&, UUID)>;

        void registerEditor(EditorType type, CreatorFunc func) {
            m_creators[type] = std::move(func);
        }

        Editor createEditor(EditorType type, VulkanRenderPassCreateInfo &createInfo, UUID uuid) {
            auto it = m_creators.find(type);
            if (it != m_creators.end()) {
                return it->second(createInfo, uuid);
            }
            Editor editor(m_defaultCreateInfo);
            return editor;
        }

    private:
        std::unordered_map<EditorType, CreatorFunc> m_creators;
        VulkanRenderPassCreateInfo& m_defaultCreateInfo;
    };

}

#endif //MULTISENSE_VIEWER_EDITORFACTORY_H
