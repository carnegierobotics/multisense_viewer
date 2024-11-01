//
// Created by mgjer on 18/08/2024.
//

#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImage.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImageLayer.h"
#include "Viewer/VkRender/Editors/Common/CommonEditorFunctions.h"

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

        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
    }

    void EditorImage::onEditorResize() {
    }

    void EditorImage::onFileDrop(const std::filesystem::path &path) {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {
            m_colorTexture = EditorUtils::createTextureFromFile(path, m_context);
            m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());
        }
    }


    void EditorImage::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();
    }


    void EditorImage::onPipelineReload() {
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
            m_context->multiSense()->getImage(&data); {
                m_multiSenseTexture->loadImage(data.imagePtr, 960 * 600);
            }
            free(data.imagePtr);
        }

        if (imageUI->update) {
            // Get offscreen rendered image
            auto sceneRenderer = m_context->getSceneRendererByUUID(getUUID());
            if (!sceneRenderer) {
                sceneRenderer = m_context->addSceneRendererWithUUID(getUUID(), m_createInfo.width, m_createInfo.height);
                auto &image = sceneRenderer->getOffscreenFramebuffer().resolvedImage;
                VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
                textureCreateInfo.image = image;
                m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
                m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());
            }
            imageUI->update = false;
        }

        if (imageUI->playVideoFromFolder) {

            std::filesystem::path folderPath = "/home/magnus/PycharmProjects/multisense-rgbd/datasets/logqs_dataset/jeep_gravel/aux_rectified";
            std::vector<std::filesystem::path> files;

            // Iterate through the folder and collect files
            for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
                if (entry.is_regular_file()) {
                    files.push_back(entry.path());
                }
            }

            // Sort files by filename assuming filenames are timestamps in nanoseconds
            std::sort(files.begin(), files.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
                return a.filename().string() < b.filename().string();
            });

            std::filesystem::path imagePath = files[m_playVideoFrameIndex];
            m_colorTexture = EditorUtils::createTextureFromFile(imagePath, m_context);
            m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());
            m_playVideoFrameIndex++;

            if (m_playVideoFrameIndex >= files.size()) {
                m_playVideoFrameIndex = 0;
            }
        }
    }

    void EditorImage::onRender(CommandBuffer &drawCmdBuffers) {
        m_renderPipelines->draw(drawCmdBuffers);
    }

    void EditorImage::onMouseMove(const MouseButtons &mouse) {
    }

    void EditorImage::onMouseScroll(float change) {
    }



}
