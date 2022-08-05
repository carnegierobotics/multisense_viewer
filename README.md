# MultiSense Viewer

[![GitHub version](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/yilber/readme-boilerplate)
[![License](https://img.shields.io/github/license/yilber/readme-boilerplate.svg)](https://github.com/Yilber/readme-boilerplate/blob/master/LICENSE)
<!---
 [![Backers on Patreon](https://img.shields.io/badge/backer-Patreon-orange.svg)](https://www.patreon.com/yilber) [![Backers on Paypal](https://img.shields.io/badge/backer-Paypal-blue.svg)](https://www.paypal.me/Yilber) -->


Boilerplate for creating github compatible readme files

## Background

> MultiSense Viewer

## Installation

### Linux/Ubuntu packages
Install Dependencies:

Open your terminal and type in
```sh
$ sudo apt update
$ sudo apt install build-essential cmake libzstd-dev ninja-build doxygen libsdl2-dev libgl1-mesa-glx libgl1-mesa-dev libvulkan1 libvulkan-dev libassimp-dev opencl-c-headers
$ sudo apt install pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libtbb-dev
```
To get the latest vulkan SDK for Ubuntu 20.04. use the following:
```sh
$ wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
$ sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.216-focal.list https://packages.lunarg.com/vulkan/1.3.216/lunarg-vulkan-1.3.216-focal.list
$ sudo apt update
$ sudo apt install vulkan-sdk
```
Clone the git repo and build the MultiSense Viewer
```sh
$ git clone --recurse-submodules -j4 https://github.com/M-Gjerde/MultiSense
$ cd MultiSense
$ mkdir build && cd build
$ cmake ..
$ make -j12 # j-parallell jobs (Same number as cores on your cpu for faster compile)
```

In your build folder there is a executable called Multisense-viewer. Launch this to start the application.
To be able to use the auto connect feature, launch with root/admin privileges.

### Windows
Prerequisites:

Have Microsoft's Visual studio installed. [Link to download](https://visualstudio.microsoft.com/vs/) <br/>
Install the Vulkan SDK for windows. Get the SDK installer [here](https://sdk.lunarg.com/sdk/download/1.3.216.0/windows/VulkanSDK-1.3.216.0-Installer.exe) and run it.

Clone this repo using git
``` sh
$ git clone --recurse-submodules -j4 https://github.com/M-Gjerde/MultiSense
```
#### Tested with Visual Studio 2022. 
1. Open up a new cmake project in Visual studio. Remember to launch as admin.
2. 




## Folder structure

Install winpcap. [link to installer](https://www.winpcap.org/install/bin/WinPcap_4_1_3.exe) Used to enable the auto connect feature.
Primarily documented using doxygen. Use a browser to view documentation located in docs/html.

```text
├── docs
│   └── html
├── src
│   └── program source**
├── LICENSE
└── README.md
```

## How to use

Explanation coming


## Bugs


## Author(s)

* [**Magnus Gjerde**](https://github.com/M-Gjerde/)

## Support

## License

Usage is provided under the MIT License. See [LICENSE](https://github.com/M-Gjerde/MultiSense/blob/master/LICENSE) for the full details.
