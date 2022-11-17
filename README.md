# MultiSense Viewer

[![GitHub version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/yilber/readme-boilerplate)
[![License](https://img.shields.io/github/license/yilber/readme-boilerplate.svg)](https://github.com/Yilber/readme-boilerplate/blob/master/LICENSE)
<!---
 [![Backers on Patreon](https://img.shields.io/badge/backer-Patreon-orange.svg)](https://www.patreon.com/yilber) [![Backers on Paypal](https://img.shields.io/badge/backer-Paypal-blue.svg)](https://www.paypal.me/Yilber) -->


| Platform | CI Status                                                                                                                                                                    |
|----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linux    | [![Linux Build Status](https://ci.appveyor.com/api/projects/status/abynv8xd75m26qo9/branch/master?svg=true)](https://ci.appveyor.com/project/ianfixes/arduino-ci)            |
| Windows  | [![Windows Build status](https://ci.appveyor.com/api/projects/status/abynv8xd75m26qo9/branch/master?svg=true)](https://travis-ci.org/Arduino-CI/arduino_ci)   |

## Background
Quickly test your MultiSense device by using this application. The application will automatically find the camera and configure your network adapter. Additionally, the app provides a rich 2D/3D viewer and options to control sensor parameters.

![Alt Text](https://github.com/M-Gjerde/MultiSense/blob/master/docs/usage_3.gif)


## Installation
### Ubuntu:
Head over to releases https://github.com/M-Gjerde/MultiSense/releases, download and install the deb package from the latest release. Compiled with gcc-9, tested on a clean Ubuntu 20.04 LTS.
The start script is located at
> /opt/multisense/start.sh

Run with root privileges to let the application configure the network for you.

### Windows
Head over to releases https://github.com/M-Gjerde/MultiSense/releases, download and install the the MultiSenseSetup.exe installer.
Run with root privileges to let the application configure the network for you.
To use the autoconnect feature also have WinPcap drivers installed, installer also located at releases page.


## How to use

[**Build Instructions -- for developers**](https://github.com/M-Gjerde/MultiSense/blob/master/docs/BUILD.md)

## Author(s)

* [**Magnus Gjerde**](https://github.com/M-Gjerde/)

## License

Usage is provided under the MIT License. See [LICENSE](https://github.com/M-Gjerde/MultiSense/blob/master/LICENSE) for the full details.
