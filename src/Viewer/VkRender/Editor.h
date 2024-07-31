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

        explicit Editor(VulkanRenderPassCreateInfo &createInfo, UUID uuid = UUID());

        // Implement move constructor
        Editor(Editor &&other) noexcept: m_context(other.m_context), m_renderUtils(other.m_renderUtils),
                                         m_renderStates(other.m_renderStates), m_createInfo(other.m_createInfo),
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
            std::swap(first.m_applicationWidth, second.m_applicationWidth);
            std::swap(first.m_applicationHeight, second.m_applicationHeight);
            std::swap(first.m_renderStates, second.m_renderStates);
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

        bool isSafeToDelete(size_t index) const { return m_renderStates[index] == RenderState::Idle; }

        void setRenderState(size_t index, RenderState state) { m_renderStates[index] = state; }

        void addUI(const std::string& layerName) { m_guiManager->pushLayer(layerName); }

        [[nodiscard]] const EditorSizeLimits &getSizeLimits() const {return m_sizeLimits;}
        VulkanRenderPassCreateInfo &getCreateInfo() { return m_createInfo; }
        [[nodiscard]] ImGuiContext *guiContext() const { return m_guiManager->m_imguiContext; }
        [[nodiscard]] UUID getUUID() const { return m_uuid; }
        EditorUI& ui() { return m_ui; }
        void setUIState(const EditorUI& state) { m_ui = state; }

        void render(CommandBuffer &drawCmdBuffers);

        void update(bool updateGraph, float frametime, Input *input);

        void updateBorderState(const glm::vec2 &mousePos);
        int32_t roundToGrid(double value, int gridSize);

        EditorBorderState checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize);

        bool validateEditorSize(VulkanRenderPassCreateInfo &createInfo);
        void resize(VulkanRenderPassCreateInfo &createInfo);

    private:
        UUID m_uuid;
        std::vector<VulkanRenderPass> m_renderPasses;
        RenderUtils &m_renderUtils;
        Renderer *m_context;
        EditorSizeLimits m_sizeLimits;

        uint32_t m_applicationWidth = 0;
        uint32_t m_applicationHeight = 0;

        std::vector<RenderState> m_renderStates;  // States for each swapchain image
        VulkanRenderPassCreateInfo m_createInfo;
        std::unique_ptr<GuiManager> m_guiManager;
        EditorUI m_ui;


    };
}

#endif //MULTISE{}NSE_VIEWER_EDITOR_H
