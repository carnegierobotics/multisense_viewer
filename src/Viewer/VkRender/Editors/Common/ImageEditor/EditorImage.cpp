//
// Created by mgjer on 18/08/2024.
//

#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImage.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/RenderPipelines/RenderBase.h"

namespace VkRender {
    EditorImage::EditorImage(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("EditorImageLayer");

        /*
        imageComponent = std::make_unique<ImageComponent>(Utils::getTexturePath() / "icon_preview.png");

        // Decide which pipeline to use
        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_renderPipelines->bindImage(*imageComponent);
        */

        /*
        // Create Texture for our image view
        VkImageCreateInfo imageCreateInfo;
        VkImageViewCreateInfo imageViewCreateInfo;

        VulkanImageCreateInfo vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCreateInfo, imageViewCreateInfo);
        std::shared_ptr<VulkanImage> image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
        */
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_context->sharedEditorData().depthFrameBuffer[m_context->sharedEditorData().selectedUUIDContext].depthImage; // TODO get this from the Editor3D Viewport then I think it is solved
        m_texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        m_depthImagePipeline = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_depthImagePipeline->bindTexture(m_texture);
        int debug = 1;
    }

    void EditorImage::onEditorResize() {
        m_recreateOnNextImageChange = true;
    }

    void EditorImage::onFileDrop(const std::filesystem::path &path) {
        /*
        if (m_recreateOnNextImageChange){
            std::filesystem::path filePath = imageComponent->getTextureFilePath();
            imageComponent = std::make_unique<ImageComponent>(filePath);
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            // Decide which pipeline to use
            m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
            m_renderPipelines->bindImage(*imageComponent);
            m_recreateOnNextImageChange = false;
        }

        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {
            imageComponent = std::make_unique<ImageComponent>(path);
            m_renderPipelines->bindImage(*imageComponent);
        }
        */
    }

    void EditorImage::onSceneLoad(std::shared_ptr<Scene> scene) {

    }


    void EditorImage::onUpdate() {
        // update Image when needed

        // Check if any 3D viewport is rendering depth

        // We want to attach the depth only image

        // Update it from the latest render
    }

    void EditorImage::onRender(CommandBuffer &drawCmdBuffers) {


        m_depthImagePipeline->draw(drawCmdBuffers);
    }

    void EditorImage::onMouseMove(const MouseButtons &mouse) {
    }

    void EditorImage::onMouseScroll(float change) {
    }

}