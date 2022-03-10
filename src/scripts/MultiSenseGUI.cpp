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
        uiSettings->sharedData = camera;

        cameraName->string = camera->getInfo().devInfo.name;
        // Generate drop downs
        modes = new DropDownItem("Type");
        for (int i = 0; i < camera->getInfo().supportedDeviceModes.size(); ++i) {
            modes->selected = "Select device mode";
            auto mode = camera->getInfo().supportedDeviceModes[i];
            std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " + std::to_string(mode.disparities) + "x";
            modes->dropDownItems.push_back(modeName);
        }
        renderUtils.ui->createDropDown(modes);

        // Start stream
        startStream = new Button("Start stream", 175.0f, 30.0f);
        renderUtils.ui->createButton(startStream);

        connectButton->clicked = false;
    }

    if (startStream != nullptr){
        if (startStream->clicked){
            camera->start(modes->selected);
            camera->play = true;
        }

    }
}


void MultiSenseGUI::draw(VkCommandBuffer commandBuffer, uint32_t i) {

}
