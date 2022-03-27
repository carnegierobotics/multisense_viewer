//
// Created by magnus on 3/1/22.
//

#include "GUISidebar.h"

void GUISidebar::setup() {

    camera = new CRLPhysicalCamera(CrlNone);
    /**
     * Create UI Elements
     */
    // UI creation

    addDevice = new Button("ADD DEVICE", 200.0f, 35.0f);
    renderUtils.ui->createButton(addDevice, "sidebar", 20, 650);

    connectButton = new Button("Connect", 175.0f, 30.0f);
    renderUtils.ui->addModalButton(connectButton);

    cameraNameHeader = new Text("Camera Name", ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    cameraNameHeader->sameLine = true;

    cameraName = new InputText("Name", 175.0f, 30.0f);
    renderUtils.ui->addModalText(cameraName);

    cameraIP = new InputText("10.66.171.21", 175.0f, 30.0f);
    renderUtils.ui->addModalText(cameraIP);
    //renderUtils.ui->createButton(connectButton, "sidebar");
    //renderUtils.ui->createText(cameraNameHeader, "sidebar");
    //renderUtils.ui->createText(cameraName, "sidebar");
}


void GUISidebar::update() {

}

void GUISidebar::onUIUpdate(UISettings *uiSettings) {
    // Creates popup under the hood
    if (addDevice->clicked){
    }

    // Add sidebar Element and attempt to connect
    if (connectButton->clicked){
        printf("Connect to: %s at %s\n", cameraName->string, cameraIP->string);
        uiSettings->closeModalPopup = true;
    }



    if (connectButton->clicked && !camera->online){
        camera->connect();
        if (!camera->online)
            return;
        uiSettings->sharedData = camera;

        // Generate drop downs
        modes = new DropDownItem("Mode");
        for (int i = 0; i < camera->getInfo().supportedDeviceModes.size(); ++i) {
            auto mode = camera->getInfo().supportedDeviceModes[i];
            std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " + std::to_string(mode.disparities) + "x";
            modes->dropDownItems.push_back(modeName);
        }
        modes->selected = "Select device mode";
        renderUtils.ui->createDropDown(modes, "main", 10, 10);

        //
        sources = new DropDownItem("Source");
        for (const auto& elem: camera->supportedSources()) {
            std::string sourceName = camera->dataSourceToString(elem);
            sources->dropDownItems.push_back(sourceName);
        }
        sources->selected = "Select data source";
        renderUtils.ui->createDropDown(sources, "main", 10, 30);

        // Start stream
        startStream = new Button("Start stream", 175.0f, 20.0f);
        renderUtils.ui->createButton(startStream, "main", 10, 55);

        // Start stream
        stopStream = new Button("Stop stream", 175.0f, 20.0f);
        renderUtils.ui->createButton(stopStream,"main", 10, 80);

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


void GUISidebar::draw(VkCommandBuffer commandBuffer, uint32_t i) {

}
