//
// Created by magnus on 3/1/22.
//

#include "MultiSenseGUI.h"

void MultiSenseGUI::setup() {

    camera = new CRLPhysicalCamera(CrlNone);
    /**
     * Create UI Elements
     */
    // UI creation
    connectButton = new Button("Connect Camera", 175.0f, 30.0f);
    renderUtils.ui->createButton(connectButton);

    cameraNameHeader = new Text("Camera Name", ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    cameraNameHeader->sameLine = true;
    renderUtils.ui->createText(cameraNameHeader);

    cameraName = new Text("No camera connected", ImVec4(0.5f, 1.0f, 1.0f, 1.0f));
    renderUtils.ui->createText(cameraName);

}


void MultiSenseGUI::update() {

}

void MultiSenseGUI::onUIUpdate(UISettings *uiSettings) {

    if (connectButton->clicked && !camera->online){
        camera->connect();
        if (!camera->online)
            return;
        uiSettings->sharedData = camera;

        cameraName->string = camera->getInfo().devInfo.name;
        // Generate drop downs
        modes = new DropDownItem("Mode");
        for (int i = 0; i < camera->getInfo().supportedDeviceModes.size(); ++i) {
            auto mode = camera->getInfo().supportedDeviceModes[i];
            std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " + std::to_string(mode.disparities) + "x";
            modes->dropDownItems.push_back(modeName);
        }
        modes->selected = "Select device mode";
        renderUtils.ui->createDropDown(modes);

        //
        sources = new DropDownItem("Source");
        for (const auto& elem: camera->supportedSources()) {
            std::string sourceName = camera->dataSourceToString(elem);
            sources->dropDownItems.push_back(sourceName);
        }
        sources->selected = "Select data source";
        renderUtils.ui->createDropDown(sources);

        // Start stream
        startStream = new Button("Start stream", 175.0f, 30.0f);
        renderUtils.ui->createButton(startStream);

        // Start stream
        stopStream = new Button("Stop stream", 175.0f, 30.0f);
        renderUtils.ui->createButton(stopStream);

        connectButton->clicked = false;
    }

    if (startStream != nullptr){
        if (startStream->clicked){
            camera->start(modes->selected, sources->selected);
            camera->play = true;
        }
    }

    if (stopStream != nullptr){
        if (stopStream->clicked){
            camera->stop(sources->selected);
            camera->play = false;
        }
    }
}


void MultiSenseGUI::draw(VkCommandBuffer commandBuffer, uint32_t i) {

}
