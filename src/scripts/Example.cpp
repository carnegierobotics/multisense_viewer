#include "Example.h"

void Example::setup() {
    printf("MyModelExample setup\n");

    std::string fileName;
    //loadFromFile(fileName);
    model.loadFromFile(Utils::getAssetsPath() + "Models/DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf", renderUtils.device,
                       renderUtils.device->transferQueue, 1.0f);


    // UI cretion
    std::vector<UISettings::DropDownItem> dropDownItems;
    dropDownItems.emplace_back(UISettings::DropDownItem("type1"));
    dropDownItems.emplace_back(UISettings::DropDownItem("type2"));
    dropDownItems.emplace_back(UISettings::DropDownItem("type3"));

    dropDownItems[0].dropdown = "Grayscale";
    dropDownItems[1].dropdown = "Albedo";
    dropDownItems[2].dropdown = "Albedo + Normal";

    for (auto item: dropDownItems) {
        renderUtils.ui->createDropDown(&item);
    }

    // Shader creation
    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/helmet.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/helmet.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    renderUtils.shaders = {{vs},
                           {fs}};

    // Obligatory call to prepare render resources for glTFModel.
    glTFModel::createRenderPipeline(renderUtils);
}

void Example::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    glTFModel::draw(commandBuffer, i);
}

void Example::update() {
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(4.0f, -5.0f, -1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;


    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (FragShaderParams *) bufferTwoData;
    d2->objectColor =  glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -2.0f, -3.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;

    bufferOneData = d;
    bufferTwoData = d2;
    bufferThreeData = selection;
}



void Example::onUIUpdate(UISettings uiSettings) {

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