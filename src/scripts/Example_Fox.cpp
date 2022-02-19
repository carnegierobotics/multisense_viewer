#include "Example_Fox.h"

void Example_Fox::setup(SetupVars vars) {

    printf("MyModelExample setup\n");
    this->vulkanDevice = vars.device;
    b.device = vars.device;
    b.UBCount = vars.UBCount;
    b.renderPass = vars.renderPass;

    std::string fileName;
    //loadFromFile(fileName);
    model.loadFromFile(Utils::getAssetsPath() + "Models/DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf", vars.device,
                       vars.device->transferQueue, 1.0f);


    std::vector<UISettings::DropDownItem> dropDownItems;
    dropDownItems.emplace_back(UISettings::DropDownItem(type));
    dropDownItems.emplace_back(UISettings::DropDownItem(type));
    dropDownItems.emplace_back(UISettings::DropDownItem(type));

    dropDownItems[0].dropdown = "Grayscale";
    dropDownItems[1].dropdown = "Albedo";
    dropDownItems[2].dropdown = "Albedo + Normal";

    for (auto item : dropDownItems){
        vars.ui->createDropDown(&item);
    }

    createUniformBuffers();
    createDescriptorSetLayout();
    createDescriptors(b.UBCount, uniformBuffers);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/fox.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/fox.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    createPipeline(*b.renderPass, shaders);

}

void Example_Fox::update() {
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(4.0f, 0.0f, -1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    auto *d = (UBOMatrix *) data.matrix;
    d->model = mat.model;

    data.selection = selection;

}

std::string Example_Fox::getType() {
    return type;
}


void Example_Fox::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    glTFModel::draw(commandBuffer, i);

}

void Example_Fox::onUIUpdate(UISettings uiSettings) {

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