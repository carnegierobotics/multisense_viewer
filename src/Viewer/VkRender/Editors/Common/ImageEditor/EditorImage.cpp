//
// Created by mgjer on 18/08/2024.
//

#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImage.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/RenderPipelines/RenderBase.h"
#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImageLayer.h"

namespace VkRender {
    EditorImage::EditorImage(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorImageLayer");
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUIData<EditorImageUI>();

        /*
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R8_UNORM;
        imageCI.extent = {static_cast<uint32_t>(960), static_cast<uint32_t>(600), 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.format = VK_FORMAT_R8_UNORM;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.debugInfo = "Color texture: Image Editor";
        m_multiSenseImage = std::make_shared<VulkanImage>(vulkanImageCreateInfo);

        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_multiSenseImage;
        m_multiSenseTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);


        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_renderPipelines->bindTexture(m_multiSenseTexture);
        */

    }

    void EditorImage::onEditorResize() {
        m_recreateOnNextImageChange = true;
    }

    void EditorImage::onFileDrop(const std::filesystem::path &path) {


        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {

            int texWidth, texHeight, texChannels;
            stbi_uc *pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            VkDeviceSize imageSize = texWidth * texHeight * 4;  // Assuming STBI_rgb_alpha gives us 4 channels per pixel

            if (!pixels) {
                throw std::runtime_error("Failed to load texture image!");
            }

            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();

            VkImageCreateInfo imageCI = Populate::imageCreateInfo();
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageCI.extent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage =
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCI.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageViewCI.subresourceRange.baseMipLevel = 0;
            imageViewCI.subresourceRange.levelCount = 1;
            imageViewCI.subresourceRange.baseArrayLayer = 0;
            imageViewCI.subresourceRange.layerCount = 1;
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

            VulkanImageCreateInfo vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCI,
                                                        imageViewCI);
            vulkanImageCreateInfo.debugInfo = "Color texture: Image Editor";
            m_colorImage = std::make_shared<VulkanImage>(vulkanImageCreateInfo);

            VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
            textureCreateInfo.image = m_colorImage;
            m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

            // Copy data to texturere
            m_colorTexture->loadImage(pixels, imageSize);
            // Free the image data
            stbi_image_free(pixels);

            m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
            m_renderPipelines->bindTexture(m_colorTexture);
        }

    }

    void EditorImage::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();
    }


    void EditorImage::onPipelineReload() {
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_context->sharedEditorData().depthFrameBuffer[m_context->sharedEditorData().selectedUUIDContext.operator*()].depthImage; // TODO get this from the Editor3D Viewport then I think it is solved
        m_texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        m_depthImagePipeline = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_depthImagePipeline->bindTexture(m_texture);
    }

    void EditorImage::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorImageUI>(m_ui);
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        if (imageUI->renderMultiSense) {
            // get image from multisense
            MultiSense::MultiSenseStreamData data;
            data.imagePtr = static_cast<uint8_t *>(malloc(960 * 600));
            data.width = 960;
            data.height = 600;
            data.dataSource = "Luma Left";
            m_context->multiSense()->getImage(&data);
            {
                m_multiSenseTexture->loadImage(data.imagePtr, 960 * 600);
            }
            free(data.imagePtr);
        }

        if (imageUI->update) {
            // Get offscreen rendered image
            auto sceneRenderer = m_context->getSceneRendererByUUID(getUUID());
            if (!sceneRenderer) {
                sceneRenderer = m_context->addSceneRendererWithUUID(getUUID());
                auto& image = sceneRenderer->getOffscreenFramebuffer().resolvedImage;
                VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
                textureCreateInfo.image = image;
                m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
                RenderPassInfo renderPassInfo{};
                renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
                renderPassInfo.renderPass = m_renderPass->getRenderPass();
                m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
                m_renderPipelines->bindTexture(m_colorTexture);
            }
            auto view = m_activeScene->getRegistry().view<CameraComponent, TagComponent>();
            std::vector<std::string> cameraEntityNames;
            cameraEntityNames.reserve(view.size_hint());
            for (auto entity : view) {
                auto& tag = view.get<TagComponent>(entity);
                if (tag.Tag == imageUI->selectedCameraName) {
                    m_ui->shared->selectedEntityMap[getUUID()] = Entity(entity, m_activeScene.get());
                }
            }
            imageUI->update = false;
        }

        if (m_context->sharedEditorData().selectedUUIDContext && !m_depthImagePipeline) {
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
            textureCreateInfo.image = m_context->sharedEditorData().depthFrameBuffer[m_context->sharedEditorData().selectedUUIDContext.operator*()].depthImage; // TODO get this from the Editor3D Viewport then I think it is solved
            m_texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
            m_depthImagePipeline = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
            m_depthImagePipeline->bindTexture(m_texture);
        }

        auto view = m_activeScene->getRegistry().view<CameraComponent>();
        for (auto entity: view) {
            auto e = Entity(entity, m_activeScene.get());
            if (e == m_createInfo.sharedUIContextData->m_selectedEntity) {
                auto &cameraComponent = view.get<CameraComponent>(entity);
                m_activeCamera = std::make_shared<Camera>(cameraComponent());
            }
        }


        if (m_depthImagePipeline && m_activeCamera) {

            m_depthImagePipeline->updateView(m_activeCamera.operator*());
            m_depthImagePipeline->update(m_context->currentFrameIndex());
        }

        if (m_renderPipelines && m_activeCamera) {
            m_renderPipelines->updateView(m_activeCamera.operator*());
            m_renderPipelines->update(m_context->currentFrameIndex());
        }
        m_activeScene->update(m_context->currentFrameIndex());
        // update Image when needed

        // Check if any 3D viewport is rendering depth

        // We want to attach the depth only image

        // Update it from the latest render
    }

    void EditorImage::onRender(CommandBuffer &drawCmdBuffers) {


        //if (m_depthImagePipeline)
        //    m_depthImagePipeline->draw(drawCmdBuffers);

        if (m_renderPipelines) {

            m_renderPipelines->draw(drawCmdBuffers);
        }
    }

    void EditorImage::onMouseMove(const MouseButtons &mouse) {
    }

    void EditorImage::onMouseScroll(float change) {
    }

}