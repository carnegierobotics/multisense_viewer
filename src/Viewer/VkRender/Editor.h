//
// Created by magnus on 7/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR_H
#define MULTISENSE_VIEWER_EDITOR_H

#include <vk_mem_alloc.h>

#include <iostream>
#include <string>

#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/EditorIncludes.h"
#include "Viewer/VkRender/Core/VulkanRenderPass.h"
#include "Viewer/VkRender/ImGui/GuiManager.h"
#include "Viewer/VkRender/Core/UUID.h"

namespace VkRender {
    class Renderer;


    class Editor {
    public:

        Editor() = delete;
        explicit Editor(VulkanRenderPassCreateInfo &createInfo, UUID uuid = UUID());

        // Implement move constructor
        Editor(Editor &&other) noexcept: m_context(other.m_context), m_renderUtils(other.m_renderUtils),
                                         m_createInfo(std::move(other.m_createInfo)),
                                         m_sizeLimits(other.m_createInfo.appWidth, other.m_createInfo.appHeight) {
            swap(*this, other);
        }

        // and move assignment operator
        Editor &operator=(Editor &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                swap(*this, other);
            }
            return *this;
        }

        // No copying allowed
        Editor(const Editor &) = delete;
        Editor &operator=(const Editor &) = delete;

        // Implement a swap function
        friend void swap(Editor &first, Editor &second) noexcept {
            std::swap(first.m_guiManager, second.m_guiManager);
            std::swap(first.m_ui, second.m_ui);
            std::swap(first.m_renderPasses, second.m_renderPasses);
            std::swap(first.m_createInfo, second.m_createInfo);
            std::swap(first.m_renderUtils, second.m_renderUtils);
            std::swap(first.m_context, second.m_context);
            std::swap(first.m_uuid, second.m_uuid);
        }

        // Comparison operator
        bool operator==(const Editor &other) const {
            return other.getUUID() == getUUID();
        }

        ~Editor() = default;

        void addUI(const std::string &layerName) { m_guiManager->pushLayer(layerName); }

        const EditorSizeLimits &getSizeLimits() const { return m_sizeLimits; }

        VulkanRenderPassCreateInfo &getCreateInfo() { return m_createInfo; }

        const VulkanRenderPassCreateInfo &getCreateInfo() const { return m_createInfo; }

        ImGuiContext *guiContext() const { return m_guiManager->m_imguiContext; }

        UUID getUUID() const { return m_uuid; }

        EditorUI &ui() { return m_ui; }

        void setUIState(const EditorUI &state) { m_ui = state; }

        void render(CommandBuffer &drawCmdBuffers);

        void update(bool updateGraph, float frametime, Input *input);

        void updateBorderState(const glm::vec2 &mousePos);

        EditorBorderState checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize);

        bool validateEditorSize(VulkanRenderPassCreateInfo &createInfo);

        void resize(VulkanRenderPassCreateInfo &createInfo);

    private:
        UUID m_uuid;
        std::vector<VulkanRenderPass> m_renderPasses;
        RenderUtils &m_renderUtils;
        Renderer *m_context;
        EditorSizeLimits m_sizeLimits;

        VulkanRenderPassCreateInfo m_createInfo;
        std::unique_ptr<GuiManager> m_guiManager;
        EditorUI m_ui;


    };
}

#endif //MULTISE{}NSE_VIEWER_EDITOR_H
