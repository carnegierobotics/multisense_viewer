//
// Created by magnus on 3/21/22.
//

#include <MultiSense/src/crl_camera/CRLVirtualCamera.h>
#include <linux/sockios.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <linux/if_ether.h>
#include "CameraConnection.h"

CameraConnection::CameraConnection() {

}

void CameraConnection::updateActiveDevice(Element dev) {

    for (auto d : dev.stream){
        if (d.playbackStatus == PREVIEW_PLAYING){

            camPtr->start(d.selectedStreamingMode, d.selectedStreamingSource);

        }
    }

    //camPtr->start(dev.selectedStreamingMode, "Color Rectified Aux");


    if (dev.button && !camPreviewBar.active) {
        camPreviewBar.active = true;
    } else if (dev.button && !camPreviewBar.active) {
        camPreviewBar.active = false;
    }


}

void CameraConnection::onUIUpdate(std::vector<Element> *devices) {
    // If no device is connected then return
    if (devices == nullptr)
        return;

    // Check for actions on each element
    for (auto &dev: *devices) {
        // Connect if we click a device or if it is just added
        if ((dev.clicked && dev.state != ArActiveState) || dev.state == ArJustAddedState) {
            connectCrlCamera(dev);
            continue;
        }

        updateDeviceState(&dev);
        if (dev.state != ArActiveState)
            continue;


        updateActiveDevice(dev);

        // Disable if we click a device already connected
        if (dev.clicked && dev.state == ArActiveState) {
            disableCrlCamera(dev);
            continue;
        }
    }


}

void CameraConnection::connectCrlCamera(Element &dev) {
    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    bool connected = false;

    if (dev.cameraName == "Virtual Camera") {
        camPtr = new CRLVirtualCamera();
        connected = camPtr->connect("None");
        if (connected) {
            dev.state = ArActiveState;
            dev.cameraName = "Virtual Camera";
            dev.IP = "Local";
            lastActiveDevice = dev.name;

        } else
            dev.state = ArUnavailableState;

    } else {
        setNetworkAdapterParameters(dev);
        camPtr = new CRLPhysicalCamera();
        connected = camPtr->connect(dev.IP);
        if (connected) {
            dev.state = ArActiveState;
            dev.cameraName = camPtr->getCameraInfo().devInfo.name;
            setStreamingModes(dev);

        } else
            dev.state = ArUnavailableState;
    }
}

void CameraConnection::setStreamingModes(Element &dev){
    // Find sources for each imager and and set these correspondly in element
    // Start with left
    dev.stream.reserve(PREVIEW_DISPARITY);

    StreamingModes left{};
    left.sources.emplace_back("None");
    left.sources.emplace_back("Raw Left");
    left.sources.emplace_back("Luma Left");
    left.sources.emplace_back("Luma Rectified Left");
    left.sources.emplace_back("Compressed");
    left.streamIndex = PREVIEW_LEFT;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";
        left.modes.emplace_back(modeName);
    }
    left.selectedStreamingMode = left.modes.front();
    left.selectedStreamingSource = left.sources.front();

    StreamingModes right{};
    right.sources.emplace_back("None");
    right.sources.emplace_back("Raw Right");
    right.sources.emplace_back("Luma Right");
    right.sources.emplace_back("Luma Rectified Right");
    right.sources.emplace_back("Compressed");
    right.streamIndex = PREVIEW_RIGHT;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        right.modes.emplace_back(modeName);

    }
    right.selectedStreamingMode = right.modes.front();
    right.selectedStreamingSource = right.sources.front();

    StreamingModes disparity{};
    disparity.sources.emplace_back("None");
    disparity.sources.emplace_back("Disparity Left");
    disparity.sources.emplace_back("Disparity Cost");
    disparity.sources.emplace_back("Disparity Cost");
    disparity.streamIndex = PREVIEW_DISPARITY;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        disparity.modes.emplace_back(modeName);

    }
    disparity.selectedStreamingMode = disparity.modes.front();
    disparity.selectedStreamingSource = disparity.sources.front();

    StreamingModes auxiliary{};
    auxiliary.sources.emplace_back("None");
    auxiliary.sources.emplace_back("Color Rectified Aux");
    auxiliary.sources.emplace_back("Luma Rectified Aux");
    auxiliary.sources.emplace_back("Color + Luma Rectified Aux");
    auxiliary.streamIndex = PREVIEW_AUXILIARY;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        auxiliary.modes.emplace_back(modeName);

    }
    auxiliary.selectedStreamingMode = auxiliary.modes.front();
    auxiliary.selectedStreamingSource = auxiliary.sources.front();

    dev.stream.emplace_back(left);
    dev.stream.emplace_back(right);
    dev.stream.emplace_back(disparity);
    dev.stream.emplace_back(auxiliary);

    lastActiveDevice = dev.name;
}

void CameraConnection::setNetworkAdapterParameters(Element &dev) {

    if ((sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) < 0) {
        fprintf(stderr, "socket SOCK_RAW: %s", strerror(errno));
    }
    // Specify interface name
    const char *interface = "enx606d3cbfbd11";
    setsockopt(sd, SOL_SOCKET, SO_BINDTODEVICE, interface, 15);

    struct ifreq ifr{};
    /// note: no pointer here
    struct sockaddr_in inet_addr{}, subnet_mask;
    /* get interface name */
    /* Prepare the struct ifreq */
    bzero(ifr.ifr_name, IFNAMSIZ);
    strncpy(ifr.ifr_name, interface, IFNAMSIZ);

    /// note: prepare the two struct sockaddr_in
    inet_addr.sin_family = AF_INET;
    int inet_addr_config_result = inet_pton(AF_INET, "10.66.171.22", &(inet_addr.sin_addr));

    subnet_mask.sin_family = AF_INET;
    int subnet_mask_config_result = inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));

    /* Call ioctl to configure network devices */
    /// put addr in ifr structure
    memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
    int ioctl_result = ioctl(sd, SIOCSIFADDR, &ifr);  // Set IP address
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFADDR: %s\n", strerror(errno));
    }

    /// put mask in ifr structure
    memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
    ioctl_result = ioctl(sd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFNETMASK: %s\n", strerror(errno));
    }

    strncpy(ifr.ifr_name, interface, sizeof(ifr.ifr_name));//interface name where you want to set the MTU
    ifr.ifr_mtu = 7200; //your MTU size here
    ioctl_result = ioctl(sd, SIOCSIFMTU, (caddr_t) &ifr);
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFMTU: %s\n", strerror(errno));
    }


}

void CameraConnection::updateDeviceState(Element *dev) {

    dev->state = ArUnavailableState;

    // IF our clicked device is the one we already clicked
    if (dev->name == lastActiveDevice) {
        dev->state = ArActiveState;
    }


}

void CameraConnection::disableCrlCamera(Element &dev) {
    dev.state = ArDisconnectedState;
    lastActiveDevice = "";


    // Free camPtr memory and point it to null for a reset.
    delete camPtr;
    camPtr = nullptr;

}


