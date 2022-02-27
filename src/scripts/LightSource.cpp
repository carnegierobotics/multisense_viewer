#include "LightSource.h"

void LightSource::setup() {
    printf("MyModelExample setup\n");

    std::string fileName;
    //loadFromFile(fileName);
    model.loadFromFile(Utils::getAssetsPath() + "Models/Box/glTF-Embedded/Box.gltf", renderUtils.device,
                       renderUtils.device->transferQueue, 1.0f);


    // UI cretion
    std::vector<UISettings::DropDownItem> dropDownItems;
    dropDownItems.emplace_back(UISettings::DropDownItem("Render"));
    dropDownItems.emplace_back(UISettings::DropDownItem("Render"));
    dropDownItems.emplace_back(UISettings::DropDownItem("Render"));

    dropDownItems[0].dropdown = "Grayscale";
    dropDownItems[1].dropdown = "Albedo";
    dropDownItems[2].dropdown = "Albedo + Normal";

    for (auto item: dropDownItems) {
        renderUtils.ui->createDropDown(&item);
    }

    // Shader creation
    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/box.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/box.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    renderUtils.shaders = {{vs},
                           {fs}};

    // Obligatory call to prepare render resources for glTFModel.
    glTFModel::createRenderPipeline(renderUtils);
}

void LightSource::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    glTFModel::draw(commandBuffer, i);
}

void LightSource::update() {
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, -3.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(0.1f, 0.1f, 0.1f));

    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

}



void LightSource::onUIUpdate(UISettings uiSettings) {

    if (uiSettings.selectedDropDown == NULL)
        return;

    if (strcmp(uiSettings.selectedDropDown, "Grayscale") == 0) {
        selection = (void *) "0";
    }
    if (strcmp(uiSettings.selectedDropDown, "Albedo") == 0) {
        selection = (void *) "1";
    }
    if (strcmp(uiSettings.selectedDropDown, "Albedo + Normal") == 0) {
        selection = (void *) "2";
    }

    printf("Selection %s\n", (char *) selection);
}