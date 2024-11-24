//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGE
#define MULTISENSE_VIEWER_EDITORIMAGE


#include "Viewer/VkRender/Editors/DescriptorRegistry.h"
#include "Viewer/VkRender/Editors/PipelineManager.h"

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/VkRender/Core/VulkanTexture.h"
#include "Viewer/VkRender/RenderResources/GraphicsPipeline2D.h"
#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/DiffRenderEntry.h"

namespace VkRender {


    class EditorImage : public Editor {
    public:
        EditorImage() = delete;

        explicit EditorImage(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;
        void collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
            uint32_t frameIndex);
        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        ~EditorImage() override = default;

        void onMouseMove(const MouseButtons &mouse) override;
        void onPipelineReload() override;

        void onFileDrop(const std::filesystem::path &path) override;

        void onMouseScroll(float change) override;

        void onEditorResize() override;

    private:
        std::vector<std::unique_ptr<Buffer>> m_shaderSelectionBuffer;
        PipelineManager m_pipelineManager;
        DescriptorRegistry m_descriptorRegistry;
        std::shared_ptr<MeshInstance> m_meshInstances;
        std::shared_ptr<VulkanTexture2D> m_colorTexture;

        std::unique_ptr<DR::DiffRenderEntry> diffRenderEntry;

    };
}

#endif //MULTISENSE_VIEWER_EDITORIMAGE
