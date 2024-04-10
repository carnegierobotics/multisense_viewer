//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/ImageViewer.h"
#include "Viewer/ImGui/Widgets.h"

void ImageViewer::setup() {
    // Create a Quad Texture
    // - Will change very often
    // - Determine where to display on the screen
    // - Very similar to CRLCameraModels but should be minimal
    uint32_t width = 300, height = 300, channels = 4;

    std::string vertexShaderFileName = "spv/default.vert";
    std::string fragmentShaderFileName = "spv/default.frag";
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};


    imageView = std::make_unique<ImageView>(&renderUtils, width, height, channels, &shaders, true);

    imageView2 = std::make_unique<ImageView>(&renderUtils, width, height, channels, &shaders, false);


}


void ImageViewer::update() {
    {
        if (renderData.scriptDrawCount < 100){
            auto &d = ubo[1].mvp;
            d->model = glm::mat4(1.0f);
            d->view = renderData.camera->matrices.view;
            d->projection = renderData.camera->matrices.perspective;

        }

    }

    auto &d = ubo[0].mvp;
    d->model = glm::mat4(1.0f);
    d->view = renderData.camera->matrices.view;
    d->projection = renderData.camera->matrices.perspective;

    float xOffsetPx = (renderData.width - 150.0) / renderData.width;

    float translationX = xOffsetPx * 2 - 1;
    float translationY = xOffsetPx * 2 - 1;

    // Apply translation after scaling
    d->model = glm::translate(d->model, glm::vec3(translationX, translationY, 0.0f));
    // Convert 300 pixels from the right edge into NDC
    float scaleX = 300.0f / renderData.width;

    d->model = glm::scale(d->model, glm::vec3(scaleX, scaleX, 1.0f)); // Uniform scaling in x and y

    *ubo[1].mvp = *ubo[0].mvp;
    //size_t size = 1920 * 1080 * 4;
    //auto* pixels = (uint8_t*) malloc(1920 * 1080 * 4);
    //for(size_t i = 0; i < size; ++i){
    //    float noise = static_cast<float>(random()) / static_cast<float>(RAND_MAX); // Generate a value between 0.0 and 1.0
    //    pixels[i] = static_cast<uint8_t>(noise * 255); // Scale to 0-255 and convert to uint8_t
    //}


    imageView2->updateTexture(renderData.index, syclRenderer.fb.data(), syclRenderer.fb.size());


    //free(pixels);
}

void ImageViewer::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {
    if (commandBuffer->renderPassIndex == 0 && b) {
        imageView->draw(commandBuffer, i);
    }
    if (b)
        imageView2->draw(commandBuffer, i);

}
