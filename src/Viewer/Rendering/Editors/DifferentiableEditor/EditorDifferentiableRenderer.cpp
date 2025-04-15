//
// Created by magnus on 1/15/25.
//

#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.h"

#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"
#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRendererLayerUI.h"

#include <OpenImageDenoise/oidn.hpp>

#include <Viewer/Rendering/Components/ScriptableComponent.h>
#include <Viewer/Scripts/Rays/GradientRay.h>

#include <yaml-cpp/yaml.h>

namespace VkRender {
    EditorDifferentiableRenderer::EditorDifferentiableRenderer(EditorCreateInfo &createInfo, UUID uuid) : Editor(
        createInfo, uuid) {
        addUI("EditorDifferentiableRendererLayer");
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUIData<EditorDifferentiableRendererLayerUI>();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());

        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());
        for (auto &frameIndex: m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                frameIndex,
                sizeof(float), nullptr, "EditorDifferentiableRenderer:ShaderSelectionBuffer",
                m_context->getDebugUtilsObjectNameFunction());
        }
        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_R8G8B8A8_UNORM, m_context);
    }

    void EditorDifferentiableRenderer::onEditorResize() {
    }


    void EditorDifferentiableRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
    }


    void EditorDifferentiableRenderer::updatePathTracerSettings() {
        auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_ui);
        auto activeCamera = m_context->activeScene()->getActiveCamera();
        Log::Logger::getInstance()->info("Setting New Kernel Device");
        SYCLDeviceType deviceType = SYCLDeviceType::GPU;
        if (imageUI->kernelDevice == "CPU") {
            deviceType = SYCLDeviceType::CPU;
        }
        auto syclDevice = m_context->getSyclDeviceSelector().getDevice(deviceType);
        uint32_t width = m_createInfo.width;
        uint32_t height = m_createInfo.height;
        if (activeCamera) {
            width = activeCamera->pinholeParameters.width;
            height = activeCamera->pinholeParameters.height;
        }
        PathTracer::PhotonTracer::PipelineSettings pipelineSettings(syclDevice, width, height);

        //std::filesystem::path ;
        std::filesystem::path datasetPath = "output/";


        std::filesystem::path filePath;
        for (const auto &entry: std::filesystem::directory_iterator(datasetPath)) {
            std::string filename = entry.path().filename().string();
            if (filename.find("render_info") != std::string::npos && filename.ends_with(".yaml")) {
                filePath = entry.path(); // Return the first matching file
            }
        }

        std::filesystem::path metricsFilePath = "metrics.csv";
        if (std::filesystem::exists(metricsFilePath))
            std::filesystem::remove(metricsFilePath.string().c_str());

        if (std::filesystem::exists(filePath)) {
            YAML::Node config = YAML::LoadFile(filePath);
            // Retrieve values from YAML nodes
            auto gamma = config["Gamma"].as<double>();
            auto photonHitCount = config["PhotonHitCount"].as<uint64_t>();
            auto photonsEmitted = config["PhotonsEmitted"].as<uint64_t>();
            auto frameCount = config["FrameCount"].as<uint32_t>();
            auto photonBounceCount = config["PhotonBounceCount"].as<uint32_t>();

            // Print them out (or use them in your application)
            std::cout << "Gamma: " << gamma << std::endl;
            std::cout << "PhotonHitCount: " << photonHitCount << std::endl;
            std::cout << "PhotonsEmitted: " << photonsEmitted << std::endl;
            std::cout << "FrameCount: " << frameCount << std::endl;
            pipelineSettings.photonCount = photonsEmitted / frameCount;
            pipelineSettings.numBounces = photonBounceCount;
            m_renderSettings.gammaCorrection = gamma;
            pipelineSettings.numFrames = frameCount;
        } else {
            Log::Logger::getInstance()->warning("Did not load params from dataset folder: {}", filePath.string());
        }

        m_pathTracer = std::make_unique<
            PathTracer::PhotonTracer>(m_context, pipelineSettings, m_context->activeScene());
        syclDevice->getQueue().wait();
        auto list = syclDevice->getQueue().get_wait_list();
        for (const auto &event: list) {
            try {
                auto status = event.get_info<sycl::info::event::command_execution_status>();
                Log::Logger::getInstance()->info("Event status: {}", static_cast<int>(status));
            } catch (const std::exception &e) {
                Log::Logger::getInstance()->error("Error retrieving event info: {}", e.what());
            }
        }
        vkDeviceWaitIdle(m_context->vkDevice().m_LogicalDevice);
        float editorAspect = static_cast<float>(m_createInfo.width) /
                             static_cast<float>(m_createInfo.height);
        float sceneCameraAspect = static_cast<float>(width) /
                                  static_cast<float>(height);
        float scaleX = 1.0f, scaleY = 1.0f;
        if (editorAspect > sceneCameraAspect) {
            scaleX = sceneCameraAspect / editorAspect;
        } else {
            scaleY = editorAspect / sceneCameraAspect;
        }
        m_meshInstances.reset();
        m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);

        m_photonRebuildModule = std::make_unique<PathTracer::PhotonRebuildModule>(
            m_pathTracer.get(), m_context->activeScene());

        m_optimizer = std::make_unique<torch::optim::SparseAdam>(
            // We pass in the parameters of our module (or custom parameter list)
            m_photonRebuildModule->parameters(),
            // Then define the Adam options, e.g. learning rate = 1e-3
            torch::optim::SparseAdamOptions().lr(0.005)
            //.betas(std::make_tuple(0.8, 0.95)) // Example
            //.eps(1e-8)
            //.maximize(false) // or true

        );
        m_accumulatedTensor = torch::Tensor();
        m_numAccumulated = 0;
        m_optimizer->zero_grad(); // Clear old gradients
    }

    void EditorDifferentiableRenderer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_ui);

        // Enable some camera if we dont have one enabled:
        if (!m_context->activeScene()->getActiveCamera()) {
            auto cameraView = m_context->activeScene()->getRegistry().view<CameraComponent>();
            std::vector<Entity> cameraEntities;
            for (auto e: cameraView) {
                auto entity = Entity(e, m_context->activeScene().get());
                if (entity.getComponent<CameraComponent>().cameraType == CameraComponent::PINHOLE) {
                    entity.getComponent<CameraComponent>().isActiveCamera() = true;
                    break;
                }
            }
        }
        if (!m_context->activeScene()->getActiveCamera())
            return;
        // 2. Check if we need to re-create the pipeline (resolution changed or user forced reset).
        if (imageUI->reloadRenderer) {
            Log::Logger::getInstance()->info("Resetting Path Tracer.. Change in settings");
            updatePathTracerSettings();
            imageUI->reloadRenderer = false;
            m_stepIteration = 0;

            // Check if the folder exists
            std::filesystem::path debugFolder = "debug";
            if (std::filesystem::exists(debugFolder) && std::filesystem::is_directory(debugFolder)) {
                for (const auto &entry: std::filesystem::directory_iterator(debugFolder)) {
                    try {
                        std::filesystem::remove_all(entry); // Removes files and subdirectories
                    } catch (const std::filesystem::filesystem_error &e) {
                        std::cerr << "Failed to remove " << entry.path() << ": " << e.what() << '\n';
                    }
                }
            }
        }


        // ----------------------------------------------------------
        // 1. Accumulate forward passes
        // ----------------------------------------------------------
        if (m_photonRebuildModule && (imageUI->step || imageUI->toggleStep)) {
            // Store camera entities (assuming there are exactly two cameras)


            CameraComponent *activeCamera = m_context->activeScene()->getActiveCamera();


            // Prepare path tracer forward settings
            // Use the actual scene camera (pinhole) from your scene
            m_renderSettings.camera = *activeCamera->getPinholeCamera();
            m_renderSettings.cameraTransform = TransformComponent(
                activeCamera->getPinholeCamera()->matrices.transform);
            if (m_previousSceneCamera != activeCamera)
                m_pathTracer->resetImage();
            m_previousSceneCamera = activeCamera;

            if (m_numAccumulated == 0) {
                m_photonRebuildModule->uploadPathTracerFromTensor(); // Upload path tracer with the new parameters
                m_photonRebuildModule->uploadSceneFromTensor(m_context->activeScene());
            }

            bool imageSizeMatch = static_cast<uint32_t>(m_renderSettings.camera.m_parameters.width) == m_colorTexture->
                                  width() &&
                                  static_cast<uint32_t>(m_renderSettings.camera.m_parameters.height) == m_colorTexture
                                  ->height();
            if (imageSizeMatch) {
                //m_renderSettings.applyBetaContribution = true;
                // Upload path tracer with the new parameters
                PathTracer::IterationInfo pathTracerIterationInfo;
                pathTracerIterationInfo.renderSettings = m_renderSettings;
                pathTracerIterationInfo.iteration = m_stepIteration;
                pathTracerIterationInfo.cameraName = m_context->activeScene()->getActiveCameraEntity().getName();
                pathTracerIterationInfo.saveDebugInfo = imageUI->saveDebugInfo;
                // Forward pass (autograd-compatible)
                m_accumulatedTensor = m_photonRebuildModule->forward(&pathTracerIterationInfo);
                m_numAccumulated++;

                // Optionally retrieve the float* for real-time display
                float *img = m_photonRebuildModule->getRenderedImage();
                uint32_t width = m_colorTexture->width();
                uint32_t height = m_colorTexture->height();
                if (!img) {
                    std::cerr << "No rendered image returned; skipping display step.\n";
                    return;
                }

                // Convert to RGBA for UI
                std::vector<uint8_t> convertedImage(width * height * 4); // RGBA
                for (uint32_t i = 0; i < width * height; ++i) {
                    float r = img[i]; // If single channel, replicate to R/G/B
                    convertedImage[i * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                    convertedImage[i * 4 + 1] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                    convertedImage[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                    convertedImage[i * 4 + 3] = 255;
                }
                m_colorTexture->loadImage(convertedImage.data(), convertedImage.size());
                Log::Logger::getInstance()->info("Forward pass no: {}/{}, Using Camera: {}", m_numAccumulated,
                                                 m_pathTracer->getPipelineSettings().numFrames,
                                                 m_context->activeScene()->getActiveCameraEntity().getName());
                // Backpropagate -- OPTIMIZATION STEP --

                if (m_numAccumulated >= m_pathTracer->getPipelineSettings().numFrames) {
                    // Load the target tensor

                    std::filesystem::path datasetPath = "output/";

                    std::filesystem::path gtFileName;

                    gtFileName = datasetPath / (m_context->activeScene()->getActiveCameraEntity().getName() + ".pfm");


                    Log::Logger::getInstance()->info("Rendered iteration: {}: gt file: {}", m_stepIteration,
                                                     gtFileName.string());
                    torch::Tensor gtTensor = loadPFM(gtFileName, width, height);


                    // Compute loss
                    auto loss = torch::mean(torch::abs(m_accumulatedTensor - gtTensor));
                    //auto loss = torch::mean(torch::pow(m_accumulatedTensor - gtTensor, 2));

                    auto start = std::chrono::high_resolution_clock::now();

                    // Backward
                    loss.backward();
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> duration = end - start;
                    std::cout << "Backward pass took " << duration.count() << " ms\n";

                    // Log loss
                    float loss_val = loss.item<float>();
                    //std::cout << "Loss: " << loss_val << std::endl;
                    Log::Logger::getInstance()->info("Loss: {}", loss_val);

                    // Calculate PSNR (assuming images are normalized to [0,1])
                    float psnr_val = 10.0f * std::log10(1.0f / loss_val);

                    // Calculate SSIM
                    float ssim_val = computeSSIM(gtTensor, m_accumulatedTensor);
                    //std::cout << "PSNR: " << psnr_val << ", SSIM: " << ssim_val << std::endl;
                    Log::Logger::getInstance()->info("PSNR: {}, SSIM: {}", psnr_val, ssim_val);

                    // Gradient checks: positions, scales, normals
                    // (Make sure you've actually registered these as parameters in your module!)
                    auto positions = m_photonRebuildModule->m_tensorData.positions;

                    auto gradPositions = m_photonRebuildModule->m_tensorData.positions.grad();
                    auto gradScales = m_photonRebuildModule->m_tensorData.scales.grad();
                    auto gradNormals = m_photonRebuildModule->m_tensorData.normals.grad();

                    auto quadricPositions = m_photonRebuildModule->m_tensorData.quadricPositions.clone();
                    auto quadricGradients = m_photonRebuildModule->m_tensorData.quadricPositions.grad().clone();

                    //m_lastIteration.positionGradient = glm::vec3(positions[0][0].item<float>(),
                    //                                             positions[0][1].item<float>(),
                    //                                             positions[0][2].item<float>());


                    // Optimizer step
                    m_optimizer->step();
                    // Reset the accumulation if you only wanted to do a single backprop per accumulation
                    m_accumulatedTensor = torch::Tensor();
                    m_numAccumulated = 0;
                    m_optimizer->zero_grad(); // Clear old gradients
                    m_stepIteration++;

                    // Save metrics to a CSV file
                    // The CSV header is: m_stepIteration, camera_id, Loss, SSIM, PSNR
                    std::ofstream csvFile;
                    // Open the file in append mode
                    csvFile.open("metrics.csv", std::ios::out | std::ios::app);
                    if (csvFile.tellp() == 0) {
                        // File is empty, so write the header
                        csvFile << "m_stepIteration, camera_id, Loss, SSIM, PSNR\n";
                    }
                    csvFile << m_stepIteration << ", "
                            << m_context->activeScene()->getActiveCameraEntity().getName() << ", "
                            << loss_val << ", "
                            << ssim_val << ", "
                            << psnr_val << "\n";
                    csvFile.close();


                    // Update the scene
                    auto &gradients = pathTracerIterationInfo.gradients;
                    auto vectors = m_context->activeScene()->getEntityByName("EntityGradients");
                    if (vectors) {
                        if (vectors.hasComponent<ScriptableComponent>()) {
                            auto &script = vectors.getComponent<ScriptableComponent>();
                            if (script.instance) {
                                auto *gradientScript = reinterpret_cast<GradientRay *>(script.instance);
                                gradientScript->ray = glm::vec3(quadricGradients[0][0].item<float>(),
                                                                 quadricGradients[0][1].item<float>(),
                                                                 quadricGradients[0][2].item<float>());
                                gradientScript->origin = glm::vec3(quadricPositions[0][0].item<float>(),
                                                                   quadricPositions[0][1].item<float>(),
                                                                   quadricPositions[0][2].item<float>());
                            }
                        }
                    }
                }
            } else {
                Log::Logger::getInstance()->warning("Image size Mismatch! Texture: {}x{}, Camera: {}x{}",
                                                    m_colorTexture->width(), m_colorTexture->height(),
                                                    m_renderSettings.camera.m_parameters.width,
                                                    m_renderSettings.camera.m_parameters.height);

                uint32_t width = m_context->activeScene()->getActiveCamera()->pinholeParameters.width;
                uint32_t height = m_context->activeScene()->getActiveCamera()->pinholeParameters.height;
                m_colorTexture = EditorUtils::createEmptyTexture(
                    width,
                    height,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    m_context);
            }
        }
    }


    float EditorDifferentiableRenderer::computeSSIM(const torch::Tensor &img1, const torch::Tensor &img2) {
        // Constants for SSIM
        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;

        // Ensure the images have 4 dimensions: {N, C, H, W}
        torch::Tensor X, Y;
        if (img1.dim() == 2) {
            X = img1.unsqueeze(0).unsqueeze(0);
            Y = img2.unsqueeze(0).unsqueeze(0);
        } else {
            X = img1;
            Y = img2;
        }

        // Use a 3x3 window for average pooling
        auto avgPoolOptions = torch::nn::functional::AvgPool2dFuncOptions(3).stride(1).padding(1);
        auto mu1 = torch::nn::functional::avg_pool2d(X, avgPoolOptions);
        auto mu2 = torch::nn::functional::avg_pool2d(Y, avgPoolOptions);
        auto mu1_sq = mu1 * mu1;
        auto mu2_sq = mu2 * mu2;
        auto mu1_mu2 = mu1 * mu2;

        auto sigma1_sq = torch::nn::functional::avg_pool2d(X * X, avgPoolOptions) - mu1_sq;
        auto sigma2_sq = torch::nn::functional::avg_pool2d(Y * Y, avgPoolOptions) - mu2_sq;
        auto sigma12 = torch::nn::functional::avg_pool2d(X * Y, avgPoolOptions) - mu1_mu2;

        auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

        return ssim_map.mean().item<float>();
    }


    void EditorDifferentiableRenderer::onRender(CommandBuffer &commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand> > renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.frameIndex);

        // Render each group
        for (auto &[pipeline, commands]: renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto &command: commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }
    }

    void EditorDifferentiableRenderer::collectRenderCommands(
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand> > &renderGroups,
        uint32_t frameIndex) {
        if (!m_meshInstances) {
            m_meshInstances = EditorUtils::setupMesh(m_context);
            Log::Logger::getInstance()->info("Created MeshInstance for 3DViewport");
        }
        if (!m_meshInstances)
            return;
        PipelineKey key = {};
        key.setLayouts.resize(1);
        auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_ui);

        // Prepare descriptor writes based on your texture or other resources
        std::array<VkWriteDescriptorSet, 2> writeDescriptors{};
        writeDescriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[0].dstBinding = 0; // Binding index
        writeDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptors[0].descriptorCount = 1;
        writeDescriptors[0].pImageInfo = &m_colorTexture->getDescriptorInfo();
        writeDescriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[1].dstBinding = 1; // Binding index
        writeDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptors[1].descriptorCount = 1;
        writeDescriptors[1].pBufferInfo = &m_shaderSelectionBuffer[frameIndex]->m_descriptorBufferInfo;
        std::vector descriptorWrites = {writeDescriptors[0], writeDescriptors[1]};
        VkDescriptorSet descriptorSet = m_descriptorRegistry.getManager(
            DescriptorManagerType::Viewport3DTexture).getOrCreateDescriptorSet(descriptorWrites);
        key.setLayouts[0] = m_descriptorRegistry.getManager(
            DescriptorManagerType::Viewport3DTexture).getDescriptorSetLayout();
        // Use default descriptor set layout
        key.vertexShaderName = "default2D.vert";
        key.fragmentShaderName = "EditorPathTracerTexture.frag";
        key.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        key.polygonMode = VK_POLYGON_MODE_FILL;
        std::vector<VkVertexInputBindingDescription> vertexInputBinding = {
            {0, sizeof(VkRender::ImageVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        };
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
        };
        key.vertexInputBindingDescriptions = vertexInputBinding;
        key.vertexInputAttributes = vertexInputAttributes;

        // Create or retrieve the pipeline
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.debugName = "EditorDifferentiableRenderer::";
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet; // Assign the descriptor set
        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void EditorDifferentiableRenderer::bindResourcesAndDraw(const CommandBuffer &commandBuffer,
                                                            RenderCommand &command) {
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        uint32_t frameIndex = commandBuffer.frameIndex;

        if (command.meshInstance->vertexBuffer) {
            VkBuffer vertexBuffers[] = {command.meshInstance->vertexBuffer->m_buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);
            // Bind index buffer if the mesh has indices
            if (command.meshInstance->indexBuffer) {
                vkCmdBindIndexBuffer(cmdBuffer, command.meshInstance->indexBuffer->m_buffer, 0,
                                     VK_INDEX_TYPE_UINT32);
            }
        }

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          command.pipeline->pipeline()->getPipeline());


        for (auto &[index, descriptorSet]: command.descriptorSets) {
            vkCmdBindDescriptorSets(
                cmdBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                command.pipeline->pipeline()->getPipelineLayout(),
                0, // TODO can't reuse the approach in SceneRenderer since we have different manager types
                1,
                &descriptorSet,
                0,
                nullptr
            );
        }

        if (command.meshInstance->indexCount > 0) {
            vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
        }
    }


    torch::Tensor EditorDifferentiableRenderer::loadPFM(const std::string &filename, int expectedWidth,
                                                        int expectedHeight) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open PFM file: " + filename);
        }

        // Read header: "PF", width, height, scale
        std::string header;
        file >> header;
        if (header != "Pf") {
            //throw std::runtime_error("Unsupported PFM format (only RGB 'Pf' supported). Got: " + header);
        }

        int width, height;
        float scale;
        file >> width >> height >> scale;
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // skip rest of line

        if (width != expectedWidth || height != expectedHeight) {
            throw std::runtime_error("PFM dimensions don't match the expected size.");
        }

        // Determine file endianness from sign of 'scale'
        bool fileIsLittleEndian = (scale < 0.f);
        float absScale = std::fabs(scale);

        // For x86, the machine is little-endian
        bool machineIsLittleEndian = true;
        bool needByteSwap = (fileIsLittleEndian != machineIsLittleEndian);

        // Allocate space (RGB => 3 channels)
        std::vector<float> data(width * height);

        // Read raw bytes
        file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
        if (!file) {
            throw std::runtime_error("Failed to read PFM pixel data.");
        }

        // Byte-swap if needed
        if (needByteSwap) {
            for (auto &px: data) {
                uint8_t *b = reinterpret_cast<uint8_t *>(&px);
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }
        }

        // Scale pixels by absScale
        for (auto &px: data) {
            px *= absScale;
        }

        // Create Torch tensor of shape [height, width, 3]
        torch::Tensor tensor2D = torch::from_blob(data.data(), {height, width}, torch::kFloat).clone();
        // Flip rows so tensor[0,:,:] is the top scanline (PFM stores bottom→top)
        tensor2D = tensor2D.flip({0});

        // If truly grayscale repeated in R/G/B, average them to get [height, width]
        //torch::Tensor tensor2D = tensor2D.mean(2);


        if (tensor2D.isnan().any().item<bool>()) {
            throw std::runtime_error("PFM data contains NaN after load!");
        }
        if (tensor2D.isinf().any().item<bool>()) {
            throw std::runtime_error("PFM data contains Inf after load!");
        }

        return tensor2D;
    }
}
