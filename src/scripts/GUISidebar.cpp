//
// Created by magnus on 3/1/22.
//

#include "GUISidebar.h"

void GUISidebar::setup() {

    /**
     * Create UI Elements
     */
    // UI creation

    addDevice = new Button("ADD DEVICE", 200.0f, 35.0f);
    renderUtils.ui->createButton(addDevice, "sidebar", 20, 650);

    connectButton = new Button("Connect", 175.0f, 30.0f);
    renderUtils.ui->addModalButton(connectButton);

    btnAddVirtualcamera = new Button("Add Virtual", 95.0f, 30.0f);
    renderUtils.ui->addModalButton(btnAddVirtualcamera);

    cameraNameHeader = new Text("Camera Name", ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
    cameraNameHeader->sameLine = true;

    cameraName = new InputText("MultiSense S30", 175.0f, 30.0f);
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
    if (addDevice->clicked) {
    }

    if (btnAddVirtualcamera->clicked){
        SidebarDevice d{};
        uiSettings->closeModalPopup = true;

        d.name = "Virtual Point Cloud";
        d.virtualCamera = new CRLVirtualCamera(CrlNone);
        d.index = sideBarElements.size();
        sideBarElements.push_back(d);
        printf("Connect to: %s\n", d.name.c_str());

    }

    // Add sidebar Element and attempt to connect
    if (connectButton->clicked) {
        printf("Connect to: %s at %s\n", cameraName->string, cameraIP->string);

        uiSettings->closeModalPopup = true;
        SidebarDevice d{};
        d.name = cameraName->string;
        d.camera = new CRLPhysicalCamera(CrlNone);

        d.index = sideBarElements.size();
        // Create new sidebar element container
        sideBarElements.push_back(d);

    }

    // Run updates for this element
    if (!sideBarElements.empty()) {
        for (auto &id: sideBarElements) {

            if (id.btnStartStream != nullptr) {
                if (id.btnStartStream->clicked) {
                    id.camera->start(id.modes->selected, id.sources->selected);
                }
            }

            if (id.btnStopStream != nullptr) {
                if (id.btnStopStream->clicked) {
                    id.camera->stop(id.sources->selected);
                    id.camera->play = false;
                }
            }

            // Create elements for this device only if camera is not connected
            if (connectButton->clicked || btnAddVirtualcamera->clicked) {
                if (id.virtualCamera != nullptr){

                    id.virtualCamera->connect(CrlPointCloud);
                    uiSettings->virtualCamera = id.virtualCamera;

                    float offset = 300.0f + (id.index * 50.0f);

                    id.cameraNameHeader = new Text(id.name);
                    renderUtils.ui->createText(id.cameraNameHeader, "sidebar", 10, offset);

                    id.connectionStatus = new Text("Connected", ImVec4(0.2f, 1.0f, 0.3f, 1.0f));
                    renderUtils.ui->createText(id.connectionStatus, "sidebar", 10, 15 + offset);
                    continue;
                }

                // Dont recreate elements
                if (id.camera->online)
                    continue;

                id.camera->connect();
                float offset = 300.0f + (id.index * 50.0f);
                uiSettings->physicalCamera = id.camera;
                id.cameraNameHeader = new Text(id.name);
                renderUtils.ui->createText(id.cameraNameHeader, "sidebar", 10, offset);

                id.connectionStatus = new Text("Connected", ImVec4(0.2f, 1.0f, 0.3f, 1.0f));
                renderUtils.ui->createText(id.connectionStatus, "sidebar", 10, 15 + offset);

                // Generate drop downs
                id.modes = new DropDownItem("Mode");
                for (int i = 0; i < id.camera->getInfo().supportedDeviceModes.size(); ++i) {
                    auto mode = id.camera->getInfo().supportedDeviceModes[i];
                    std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                                           std::to_string(mode.disparities) + "x";
                    id.modes->dropDownItems.push_back(modeName);
                }

                id.modes->selected = "Select device mode";
                renderUtils.ui->createDropDown(id.modes, "main", 10, 20);

                id.sources = new DropDownItem("Source");
                for (const auto &elem: id.camera->supportedSources()) {
                    std::string sourceName = id.camera->dataSourceToString(elem);
                    id.sources->dropDownItems.push_back(sourceName);
                }
                id.sources->selected = "Select data source";
                renderUtils.ui->createDropDown(id.sources, "main", 10, 45);

                // Start stream
                id.btnStartStream = new Button("Start stream", 175.0f, 20.0f);
                renderUtils.ui->createButton(id.btnStartStream, "main", 10, 70);

                // Start stream
                id.btnStopStream = new Button("Stop stream", 175.0f, 20.0f);
                renderUtils.ui->createButton(id.btnStopStream, "main", 10, 95);

                // Start stream
                id.btnViewPointCloud = new Button("View point cloud", 175.0f, 20.0f);
                renderUtils.ui->createButton(id.btnViewPointCloud, "main", 10, 125);

            }
        }
    }
}


void GUISidebar::draw(VkCommandBuffer commandBuffer, uint32_t i) {

}
