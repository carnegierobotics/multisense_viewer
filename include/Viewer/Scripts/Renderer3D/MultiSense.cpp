//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/MultiSense.h"

void MultiSense::setup() {



    // Load the gltf vertices/indices into vulkan
    // Load a material info
    // Load scene info
    //skybox = std::make_shared<VkRender::GLTF::Skybox>(Utils::getAssetsPath() / "Models" / "box.gltf", renderUtils.device);

    // Load vulkan render resources for gltf model
    // pipelines
    // bind renderpasses
    // Also make sure to push resources to cleanup queue if we resize or exi
    //RenderResource::GLTFModel<VkRender::GLTF::Skybox> rrSkybox(&renderUtils, skybox);

    //rrSkybox.getComponent<VkRender::GLTF::Skybox>()->draw();

}


void MultiSense::update() {

}

void MultiSense::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {


}
