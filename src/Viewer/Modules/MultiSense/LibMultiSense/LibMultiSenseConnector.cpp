/**
 * @file: MultiSense-Viewer/src/CRLCamera/MultiSense.cpp
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-1-3, mgjerde@carnegierobotics.com, Created file.
 **/
#ifdef WIN32
#define _USE_MATH_DEFINES

#include <cmath>

#endif

#include <vulkan/vulkan_core.h>

#include <cmath>

#include "LibMultiSenseConnector.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender::MultiSense {

    void LibMultiSenseConnector::connect(std::string ip, std::string adapterName) {
        Log::Logger::getInstance()->info("Attempting to connect to ip: {}. Adapter is: {}", ip, adapterName);
        std::scoped_lock lock(m_channelMutex);
        m_channel = std::make_unique<ChannelWrapper>(ip);
        if (m_channel->ptr() != nullptr) {
            crl::multisense::system::DeviceInfo devInfo;
            m_channel->ptr()->getDeviceInfo(devInfo);
            Log::Logger::getInstance()->info("We got a connection! Device info: {}, {}", devInfo.name,
                                             devInfo.buildDate);
            updateCameraInfo();
            addCallbacks(static_cast<crl::multisense::RemoteHeadChannel>(0));
            setMtu(7200, static_cast<crl::multisense::RemoteHeadChannel>(0));
            state = MULTISENSE_CONNECTED;
        } else {
            Log::Logger::getInstance()->warning("Connection to: {}. failed. Adapter is: {}", ip, adapterName);
            m_channel = nullptr;
            state = MULTISENSE_UNAVAILABLE;

        }
    }

    void LibMultiSenseConnector::update() {

    }

    void LibMultiSenseConnector::disconnect() {
        Log::Logger::getInstance()->info("Disconnecting MultiSense channel");
        std::scoped_lock lock(m_channelMutex);
        m_channel.reset();
        m_channel = nullptr;
        state = MULTISENSE_DISCONNECTED;
    }

    bool LibMultiSenseConnector::start(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
        /*
        crl::multisense::DataSource source = Utils::stringToDataSource(dataSourceStr);
        if (source == false)
            return false;
        // Start stream
        crl::multisense::Status status = channelMap[channelID]->ptr()->startStreams(source);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Started stream: {} on channel {}",
                                             Utils::dataSourceToString(source).c_str(), channelID);
            return true;
        } else
            Log::Logger::getInstance()->info("Failed to start stream: {}  status code {}",
                                             Utils::dataSourceToString(source).c_str(), status);
        return false;
    */
        return false;

    }

    bool LibMultiSenseConnector::stop(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
        /*
        if (channelMap[channelID] == nullptr)
            return false;

        crl::multisense::DataSource src = Utils::stringToDataSource(dataSourceStr);
        if (!src) {
            Log::Logger::getInstance()->info("Failed to recognize '{}' source", dataSourceStr.c_str());
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->stopStreams(src);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Stopped camera stream {} on channel {}", dataSourceStr.c_str(),
                                             channelID);
            return true;
        } else {
            Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
            return false;
        }
         */
        return false;
    }

    void LibMultiSenseConnector::remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP) {
        auto p = reinterpret_cast<ChannelWrapper *>(userDataP);
        //Log::Logger::getInstance()->info("Received m_Image {}x{} on channel {} with source {}", header.m_Width, header.m_Height, p->imageBuffer->id, header.source);
        p->imageBuffer->updateImageBuffer(std::make_shared<ImageBufferWrapper>(p->ptr(), header));
    }

    void LibMultiSenseConnector::imuCallback(const crl::multisense::imu::Header &header,
                                             void *userDataP) {
        auto p = reinterpret_cast<ChannelWrapper *>(userDataP);
        p->imuBuffer->updateIMUBuffer(std::make_shared<IMUBufferWrapper>(p->ptr(), header));

    }

    void LibMultiSenseConnector::addCallbacks(crl::multisense::RemoteHeadChannel channelID) {
        for (const auto &e: m_channelInfo.supportedDeviceModes)
            m_channelInfo.supportedSources |= e.supportedDataSources;
        // reserve double_buffers for each stream
        if (m_channel->ptr()->addIsolatedCallback(imuCallback, m_channel.get()) ==
            crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Added imu callback for channel {}", channelID);
        } else {
            Log::Logger::getInstance()->info("Failed to add callback for channel {}", channelID);
        }

        if (m_channel->ptr()->addIsolatedCallback(remoteHeadCallback,
                                                  m_channelInfo.supportedSources,
                                                  m_channel.get()) ==
            crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Added image callback for channel {}", channelID);
        } else
            Log::Logger::getInstance()->info("Failed to add callback for channel {}", channelID);
    }

    /*
      LibMultiSenseConnector::CameraInfo LibMultiSenseConnector::getCameraInfo(crl::multisense::RemoteHeadChannel idx) const {
          auto it = infoMap.find(idx);
          if (it != infoMap.end())
              return it->second;
          else {
              Log::Logger::getInstance()->traceWithFrequency("getCameraInfoTag", 1000,
                                                             "Camera info for channel {} does not exist", idx);
              return {};
          }
      }



      void LibMultiSenseConnector::getIMUConfig(crl::multisense::RemoteHeadChannel channelID) {
          crl::multisense::Status status;

          std::vector<crl::multisense::imu::Info> sensorInfos;
          std::vector<crl::multisense::imu::Config> sensorConfigs;
          uint32_t sensorSamplesPerMessage = 0;
          uint32_t sensorMaxSamplesPerMessage = 0;

          auto *channel = channelMap.find(channelID)->second->ptr();

          status = channel->getImuInfo(sensorMaxSamplesPerMessage,
                                       sensorInfos);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Failed to query imu info: {}",
                                                  crl::multisense::Channel::statusString(status));
              return;
          }
          status = channel->getImuConfig(sensorSamplesPerMessage, sensorConfigs);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Failed to query imu config: {}",
                                                  crl::multisense::Channel::statusString(status));
              return;
          }

          for (const auto &info: sensorInfos) {
              Log::Logger::getInstance()->info("Got IMU info: {}, device name: {}, units: {}", info.name, info.device,
                                               info.units);
          }

          infoMap[channelID].imuSensorInfos = sensorInfos;
          infoMap[channelID].imuSensorConfigs = sensorConfigs;

          size_t maxVal = 0;
          for (const auto &conf: sensorInfos) {
              if (conf.rates.size() > maxVal) {
                  maxVal = conf.rates.size();
              }
          }
          infoMap[channelID].imuMaxTableIndex = static_cast<uint32_t>(maxVal);
          infoMap[channelID].hasIMUSensor = true;
      }


      bool LibMultiSenseConnector::setGamma(float gamma, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);
          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error("Attempted to set gamma on a channel that was not connected, Channel {}",
                                                channelID);
              return false;
          }
          crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->info("Unable to query gamma configuration");
              return false;
          }
          infoMap[channelID].imgConf.setGamma(gamma);
          status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->info("Unable to set gamma configuration");
              return false;
          } else {
              Log::Logger::getInstance()->info("Set Gamma on channel {}", channelID);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(150));

          if (crl::multisense::Status_Ok !=
              channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
              Log::Logger::getInstance()->info("Failed to verify Gamma");
              return false;
          }
          return true;
      }

      bool LibMultiSenseConnector::setFps(float fps, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);
          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error("Attempted to set fps on a channel that was not connected, Channel {}",
                                                channelID);
              return false;
          }

          crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->info("Unable to query m_Image configuration");
              return false;
          }
          infoMap[channelID].imgConf.setFps(fps);

          status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->info("Unable to set m_Image configuration");
              return false;
          } else {
              Log::Logger::getInstance()->info("Set framerate on channel {}", channelID);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(150));


          if (crl::multisense::Status_Ok !=
              channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
              Log::Logger::getInstance()->info("Failed to verify fps");
              return false;
          }

          return true;
      }

      bool LibMultiSenseConnector::setGain(float gain, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);
          crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);

          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error("Attempted to set gain on a channel that was not connected, Channel {}",
                                                channelID);
              return false;
          }

          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->info("Unable to query image configuration");
              return false;
          }

          infoMap[channelID].imgConf.setGain(gain);
          status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);

          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->info("Unable to set image configuration");
              return false;
          } else {
              Log::Logger::getInstance()->info("Set gain on channel {}", channelID);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(150));


          if (crl::multisense::Status_Ok !=
              channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
              Log::Logger::getInstance()->info("Failed to update Gain");
              return false;
          }
          return true;
      }




      bool LibMultiSenseConnector::setPostFilterStrength(float filter, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);

          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error(
                      "Attempted to set post filter strength on a channel that was not connected, Channel {}", channelID);
              return false;
          }

          crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Unable to query m_Image configuration");
              return false;
          }
          infoMap[channelID].imgConf.setStereoPostFilterStrength(filter);
          status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Unable to set m_Image configuration");
              return false;
          } else {
              Log::Logger::getInstance()->info("Successfully set stereo post filter strength to {} on channel {}", filter,
                                               channelID);
          }

          std::this_thread::sleep_for(std::chrono::milliseconds(150));

          // Since we can only have one stereo setting among several remote head update all the infoMaps.
          for (const auto &channel: channelMap) {
              // don't bother for the VPB
              if (channel.first == crl::multisense::Remote_Head_VPB || channel.second->ptr() == nullptr)
                  continue;
              if (crl::multisense::Status_Ok != channel.second->ptr()->getImageConfig(infoMap[channel.first].imgConf)) {
                  Log::Logger::getInstance()->warning("Failed to verify fps");
                  return false;
              }
          }
          return true;
      }

      bool LibMultiSenseConnector::setHDR(bool hdr, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);

          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error("Attempted to set hdr on a channel that was not connected, Channel {}",
                                                channelID);
              return false;
          }
          crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Unable to query hdr m_Image configuration");
              return false;
          }

          infoMap[channelID].imgConf.setHdr(hdr);
          status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Unable to set hdr configuration");
              return false;
          }

          std::this_thread::sleep_for(std::chrono::milliseconds(150));

          if (crl::multisense::Status_Ok !=
              channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
              Log::Logger::getInstance()->warning("Failed to verifiy HDR");
              return false;
          }
          return true;
      }


      bool LibMultiSenseConnector::setLighting(LightingParams param, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);
          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error(
                      "Attempted to set light configuration on a channel that was not connected, Channel {}", channelID);
              return false;
          }
          crl::multisense::Status status = channelMap[channelID]->ptr()->getLightingConfig(
                  infoMap[channelID].lightConf);

          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Unable to query m_Image configuration");
              return false;
          }

          if (param.selection != -1) {
              infoMap[channelID].lightConf.setDutyCycle(param.selection, param.dutyCycle);
          } else
              infoMap[channelID].lightConf.setDutyCycle(param.dutyCycle);

          infoMap[channelID].lightConf.setNumberOfPulses(
                  static_cast<uint32_t> (param.numLightPulses *
                                         1000));  // convert from ms to us and from float to uin32_t
          infoMap[channelID].lightConf.setStartupTime(
                  static_cast<uint32_t> ((param.startupTime * 1000))); // convert from ms to us and from float to uin32_t
          infoMap[channelID].lightConf.setFlash(param.flashing);
          status = channelMap[channelID]->ptr()->setLightingConfig(infoMap[channelID].lightConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->warning("Unable to set m_Image configuration");
              return false;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(150));

          if (crl::multisense::Status_Ok !=
              channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf)) {
              Log::Logger::getInstance()->warning("Failed to update '{}' ", "Lightning");
              return false;
          }

          return true;
      }

      bool LibMultiSenseConnector::setAuxImageConfig(AUXConfig auxConfig, crl::multisense::RemoteHeadChannel channelID) {
          std::scoped_lock<std::mutex> lock(m_channelMutex);

          if (channelMap[channelID]->ptr() == nullptr) {
              Log::Logger::getInstance()->error(
                      "Attempted to set exposure on a channel that was not connected, Channel {}", channelID);
              return false;
          }

          crl::multisense::Status status = channelMap[channelID]->ptr()->getAuxImageConfig(infoMap[channelID].auxImgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->error("Unable to query aux image configuration");
              return false;
          }
          auto &p = infoMap[channelID].auxImgConf;

          p.setAutoExposure(auxConfig.ep.autoExposure);
          p.setAutoExposureMax(auxConfig.ep.autoExposureMax);
          p.setAutoExposureDecay(auxConfig.ep.autoExposureDecay);
          p.setAutoExposureTargetIntensity(auxConfig.ep.autoExposureTargetIntensity);
          p.setAutoExposureThresh(auxConfig.ep.autoExposureThresh);
          p.setAutoExposureRoi(auxConfig.ep.autoExposureRoiX, auxConfig.ep.autoExposureRoiY,
                               auxConfig.ep.autoExposureRoiWidth,
                               auxConfig.ep.autoExposureRoiHeight);
          p.setExposure(auxConfig.ep.exposure);


          p.setWhiteBalance(auxConfig.whiteBalanceRed, auxConfig.whiteBalanceBlue);
          p.setAutoWhiteBalance(auxConfig.whiteBalanceAuto);
          p.setAutoWhiteBalanceDecay(auxConfig.whiteBalanceDecay);
          p.setAutoWhiteBalanceThresh(auxConfig.whiteBalanceThreshold);
          p.setGain(auxConfig.gain);
          p.setGamma(auxConfig.gamma);
          p.setSharpeningLimit(static_cast<uint8_t>(auxConfig.sharpeningLimit));
          p.setSharpeningPercentage(auxConfig.sharpeningPercentage);
          p.enableSharpening(auxConfig.sharpening);

          status = channelMap[channelID]->ptr()->setAuxImageConfig(infoMap[channelID].auxImgConf);
          if (crl::multisense::Status_Ok != status) {
              Log::Logger::getInstance()->error("Unable to set aux image configuration");
              return false;
          } else {
              Log::Logger::getInstance()->info("Set aux image conf on channel {}", channelID);
          }

          std::this_thread::sleep_for(std::chrono::milliseconds(300));

          if (crl::multisense::Status_Ok !=
              channelMap[channelID]->ptr()->getAuxImageConfig(infoMap[channelID].auxImgConf)) {
              Log::Logger::getInstance()->error("Failed to verify aux img conf");
              return false;
          }
          return true;
      }
    */
    bool LibMultiSenseConnector::setMtu(uint32_t mtu, crl::multisense::RemoteHeadChannel channelID) {
        if (m_channel->ptr() == nullptr) {
            Log::Logger::getInstance()->error("Attempted to set mtu on a channel that was not connected, Channel {}",
                                              channelID);
            return false;
        }
        int status = m_channel->ptr()->setMtu(static_cast<int32_t> (mtu));
        if (status != crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Failed to set MTU {}", mtu);
            return false;
        } else {
            Log::Logger::getInstance()->info("Set MTU to {}", mtu);
            return true;
        }
    }

    void LibMultiSenseConnector::updateCameraInfo() {
        // The calling function will repeat this loop until everything succeeds.
        bool allSucceeded = true;
        bool interruptConnection = false;
        Log::Logger::getInstance()->trace("Updating Camera info");

        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getImageConfig(data); },
                                     m_channelInfo.imgConf,
                                     "imgConf");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getDeviceModes(data); },
                                     m_channelInfo.supportedDeviceModes, "supportedDeviceModes");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getEnabledStreams(data); },
                                     m_channelInfo.supportedSources, "supportedSources");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getNetworkConfig(data); },
                                     m_channelInfo.netConfig,
                                     "netConfig");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getVersionInfo(data); },
                                     m_channelInfo.versionInfo,
                                     "versionInfo");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getDeviceInfo(data); },
                                     m_channelInfo.devInfo,
                                     "devInfo");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getMtu(data); },
                                     m_channelInfo.sensorMTU,
                                     "sensorMTU");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getLightingConfig(data); },
                                     m_channelInfo.lightConf,
                                     "lightConf");
        if (interruptConnection) return;
        allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getImageCalibration(data); },
                                     m_channelInfo.calibration, "calibration");
        if (interruptConnection) return;

        // TODO I want to update most info on startup but this function varies. On KS21 this will always fail.
        //  This also assumes that getDeviceInfo above also succeeded
        if (m_channelInfo.devInfo.hardwareRevision !=
            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21) {
            allSucceeded &= updateAndLog([&](auto &data) { return m_channel->ptr()->getAuxImageConfig(data); },
                                         m_channelInfo.auxImgConf, "auxImageConfig");
        }

        updateQMatrix();

        if (!allSucceeded) {
            Log::Logger::getInstance()->error(
                    "Querying the camera for config did not succeed, viewer may not work correctly");
        } else {
            Log::Logger::getInstance()->info("Querying the camera for config did succeeded");
        }

        // Also check for IMU information. This is not present on most devices.
        //if (interruptConnection) return false;
        //getIMUConfig(m_channelID);
        //return allSucceeded;
    }

    /*
       bool LibMultiSenseConnector::getStatus(crl::multisense::RemoteHeadChannel m_channelID,
                                  crl::multisense::system::StatusMessage *msg) {
           std::scoped_lock<std::mutex> lock(m_m_channelMutex);
           if (m_channelMap[m_channelID]->ptr() == nullptr)
               return false;

           crl::multisense::Status status = m_channelMap[m_channelID]->ptr()->getDeviceStatus(*msg);
           return status == crl::multisense::Status_Ok;
       }


       bool LibMultiSenseConnector::setSensorCalibration(const std::string &intrinsicsFile, const std::string &extrinsicsFile,
                                             crl::multisense::RemoteHeadChannel m_channelID) {
           std::scoped_lock<std::mutex> lock(m_m_channelMutex);
           if (m_channelMap[m_channelID]->ptr() == nullptr) {
               Log::Logger::getInstance()->error(
                       "Attempted to set sensor calibration on a m_channel that was not connected, Channel {}", m_channelID);
               return false;
           }
           crl::multisense::Status status = m_channelMap[m_channelID]->ptr()->getImageCalibration(
                   infoMap[m_channelID].calibration);
           if (crl::multisense::Status_Ok != status) {
               Log::Logger::getInstance()->info("Unable to query calibration");
               return false;
           }
           bool hasAuxCamera = infoMap[m_channelID].devInfo.hardwareRevision ==
                               crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                               infoMap[m_channelID].devInfo.hardwareRevision ==
                               crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                               infoMap[m_channelID].devInfo.hardwareRevision ==
                               crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM;
           std::ifstream inFile, exFile;
           std::map<std::string, std::vector<float> > data;
           inFile.open(intrinsicsFile.c_str());
           if (!inFile) {
               Log::Logger::getInstance()->error("Failed to open intrinsics file: {}.", intrinsicsFile);
               return false;
           }
           parseYaml(inFile, data);
           inFile.close();
           if (data["M1"].size() != 3 * 3 ||
               (data["D1"].size() != 5 && data["D1"].size() != 8) ||
               data["M2"].size() != 3 * 3 ||
               (data["D2"].size() != 5 && data["D2"].size() != 8) ||
               (hasAuxCamera && data["M3"].size() != 3 * 3) ||
               (hasAuxCamera && data["D3"].size() != 5 && data["D3"].size() != 8)) {
               Log::Logger::getInstance()->error("Intrinsics file error: {}. File not complete", intrinsicsFile);
               return false;
           }
           exFile.open(extrinsicsFile.c_str());
           if (!exFile) {
               Log::Logger::getInstance()->error("Failed to open extrinsics file: {}.", extrinsicsFile);
               return false;
           }
           parseYaml(exFile, data);
           exFile.close();
           if (data["R1"].size() != 3 * 3 ||
               data["P1"].size() != 3 * 4 ||
               data["R2"].size() != 3 * 3 ||
               data["P2"].size() != 3 * 4 ||
               (hasAuxCamera && (data["R3"].size() != 3 * 3 || data["P3"].size() != 3 * 4))) {
               Log::Logger::getInstance()->error("Extrinsics file error: {}. File not complete", intrinsicsFile);
               return false;
           }

           crl::multisense::image::Calibration calibration{};

           memcpy(&calibration.left.M[0][0], &data["M1"].front(), data["M1"].size() * sizeof(float));
           memset(&calibration.left.D[0], 0, sizeof(calibration.left.D));
           memcpy(&calibration.left.D[0], &data["D1"].front(), data["D1"].size() * sizeof(float));
           memcpy(&calibration.left.R[0][0], &data["R1"].front(), data["R1"].size() * sizeof(float));
           memcpy(&calibration.left.P[0][0], &data["P1"].front(), data["P1"].size() * sizeof(float));

           memcpy(&calibration.right.M[0][0], &data["M2"].front(), data["M2"].size() * sizeof(float));
           memset(&calibration.right.D[0], 0, sizeof(calibration.right.D));
           memcpy(&calibration.right.D[0], &data["D2"].front(), data["D2"].size() * sizeof(float));
           memcpy(&calibration.right.R[0][0], &data["R2"].front(), data["R2"].size() * sizeof(float));
           memcpy(&calibration.right.P[0][0], &data["P2"].front(), data["P2"].size() * sizeof(float));

           if (hasAuxCamera) {
               memcpy(&calibration.aux.M[0][0], &data["M3"].front(), data["M3"].size() * sizeof(float));
               memset(&calibration.aux.D[0], 0, sizeof(calibration.aux.D));
               memcpy(&calibration.aux.D[0], &data["D3"].front(), data["D3"].size() * sizeof(float));
               memcpy(&calibration.aux.R[0][0], &data["R3"].front(), data["R3"].size() * sizeof(float));
               memcpy(&calibration.aux.P[0][0], &data["P3"].front(), data["P3"].size() * sizeof(float));
           }


           status = m_channelMap[m_channelID]->ptr()->setImageCalibration(calibration);
           if (crl::multisense::Status_Ok != status) {
               Log::Logger::getInstance()->error("Unable to set calibration");
               return false;
           } else {
               Log::Logger::getInstance()->info("Successfully set new calibration");

           }

           std::this_thread::sleep_for(std::chrono::milliseconds(150));

           if (crl::multisense::Status_Ok !=
               m_channelMap[m_channelID]->ptr()->getImageCalibration(infoMap[m_channelID].calibration)) {
               Log::Logger::getInstance()->info("Failed to query image calibration");
               return false;
           }
           return true;
       }


       std::ostream &
       LibMultiSenseConnector::writeImageIntrinics(std::ostream &stream, crl::multisense::image::Calibration const &calibration,
                                       bool hasAuxCamera) {
           stream << "%YAML:1.0\n";
           writeMatrix(stream, "M1", 3, 3, &calibration.left.M[0][0]);
           writeMatrix(stream, "D1", 1, 8, &calibration.left.D[0]);
           writeMatrix(stream, "M2", 3, 3, &calibration.right.M[0][0]);
           writeMatrix(stream, "D2", 1, 8, &calibration.right.D[0]);

           if (hasAuxCamera) {
               writeMatrix(stream, "M3", 3, 3, &calibration.aux.M[0][0]);
               writeMatrix(stream, "D3", 1, 8, &calibration.aux.D[0]);
           }
           return stream;
       }


       std::ostream &
       LibMultiSenseConnector::writeImageExtrinics(std::ostream &stream, crl::multisense::image::Calibration const &calibration,
                                       bool hasAuxCamera) {
           stream << "%YAML:1.0\n";
           writeMatrix(stream, "R1", 3, 3, &calibration.left.R[0][0]);
           writeMatrix(stream, "P1", 3, 4, &calibration.left.P[0][0]);
           writeMatrix(stream, "R2", 3, 3, &calibration.right.R[0][0]);
           writeMatrix(stream, "P2", 3, 4, &calibration.right.P[0][0]);

           if (hasAuxCamera) {
               writeMatrix(stream, "R3", 3, 3, &calibration.aux.R[0][0]);
               writeMatrix(stream, "P3", 3, 4, &calibration.aux.P[0][0]);
           }
           return stream;
       }


       bool LibMultiSenseConnector::saveSensorCalibration(const std::string &savePath,
                                              crl::multisense::RemoteHeadChannel m_channelID) {
           std::scoped_lock<std::mutex> lock(m_m_channelMutex);

           if (m_channelMap[m_channelID]->ptr() == nullptr) {
               Log::Logger::getInstance()->error(
                       "Attempted to save calibration from a m_channel that was not connected, Channel {}", m_channelID);
               return false;
           }

           if (crl::multisense::Status_Ok !=
               m_channelMap[m_channelID]->ptr()->getImageCalibration(infoMap[m_channelID].calibration)) {
               Log::Logger::getInstance()->info("Failed to query image calibration");
               return false;
           }

           bool hasAuxCamera = infoMap[m_channelID].devInfo.hardwareRevision ==
                               crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                               infoMap[m_channelID].devInfo.hardwareRevision ==
                               crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                               infoMap[m_channelID].devInfo.hardwareRevision ==
                               crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM;
           std::ofstream inFile, exFile;
           std::string intrinsicsFile = savePath + "/" + infoMap[m_channelID].devInfo.serialNumber + "_intrinsics.yml";
           std::string extrinsicsFile = savePath + "/" + infoMap[m_channelID].devInfo.serialNumber + "_extrinsics.yml";

           inFile.open(intrinsicsFile.c_str(), std::ios_base::out | std::ios_base::trunc);

           if (!inFile) {
               Log::Logger::getInstance()->error("Failed to open Intrinsics file for writing!");
               return false;
           }

           exFile.open(extrinsicsFile.c_str(), std::ios_base::out | std::ios_base::trunc);

           if (!exFile) {
               Log::Logger::getInstance()->error("Failed to open Extrinsics file for writing!");
               return false;
           }

           Log::Logger::getInstance()->trace("Writing intrinsics to file");
           writeImageIntrinics(inFile, infoMap[m_channelID].calibration, hasAuxCamera);
           Log::Logger::getInstance()->trace("Writing extrinsics to file");
           writeImageExtrinics(exFile, infoMap[m_channelID].calibration, hasAuxCamera);
           inFile.flush();
           exFile.flush();

           Log::Logger::getInstance()->info("Wrote camera calibration to file. HasAux: {}", hasAuxCamera);
           return true;
       }


    bool LibMultiSenseConnector::getIMUData(crl::multisense::RemoteHeadChannel m_channelID,
                                std::vector<LibMultiSenseConnector::ImuData> *gyro,
                                std::vector<LibMultiSenseConnector::ImuData> *accel) const {
        auto it = m_channelMap.find(m_channelID);
        if (it == m_channelMap.end()) {
            return false;
        }
        if (gyro == nullptr || accel == nullptr) {
            Log::Logger::getInstance()->trace("gyro or accel vectors cannot be nullptrs");
            return false;
        }
        auto header = it->second->imuBuffer->getIMUBuffer(m_channelID);
        if (header == nullptr) {
            return false;
        }

        double time = 0, prevTime = 0;
        for (auto iterator = header->data().samples.begin();
             iterator != header->data().samples.end(); ++iterator) {
            const crl::multisense::imu::Sample &s = *iterator;

            switch (s.type) {
                case crl::multisense::imu::Sample::Type_Accelerometer:
                    accel->push_back({s.x, s.y, s.z, s.time(), 0});
                    break;
                case crl::multisense::imu::Sample::Type_Gyroscope: {
                    time = s.time();
                    double dt = time - prevTime;

                    if (prevTime == 0)
                        dt = 0;

                    gyro->push_back({s.x, s.y, s.z, s.time(), dt});
                    prevTime = time;
                }
                    break;
                case crl::multisense::imu::Sample::Type_Magnetometer:
                    break;
            }

        }
        return true;
    }

    */
    void LibMultiSenseConnector::updateQMatrix() {
        /// Also update correct Q matrix with the new width
        // From LibMultisenseUtility

        m_channel->ptr()->getImageConfig(m_channelInfo.imgConf);
        crl::multisense::image::Config c = m_channelInfo.imgConf;

        float scale = ((static_cast<float>(m_channelInfo.devInfo.imagerWidth) /
                        static_cast<float>(m_channelInfo.imgConf.width())));
        float dcx = (m_channelInfo.calibration.right.P[0][2] - m_channelInfo.calibration.left.P[0][2]) *
                    (1.0f / scale);
        const float &fx = c.fx();
        const float &fy = c.fy();
        const float &cx = c.cx();
        const float &cy = c.cy();
        const float &tx = c.tx();
        // glm::mat4 indexing
        // [column][row]
        // Inserted values col by col
        glm::mat4 Q(0.0f);

        Q[0][0] = fy * tx;
        Q[1][1] = fx * tx;
        Q[2][3] = -fy;
        Q[3][0] = -fy * cx * tx;
        Q[3][1] = -fx * cy * tx;
        Q[3][2] = fx * fy * tx;
        Q[3][3] = fy * dcx;

        // keep as is
        m_channelInfo.QMat = Q;
        m_channelInfo.focalLength = fx;
        m_channelInfo.pointCloudScale = scale;
        /// Finished updating Q matrix

        auto aux = m_channelInfo.calibration.aux;
        glm::mat4 K(1.0f);
        K[0][0] = fx;
        K[1][1] = fy;
        K[2][0] = cx;
        K[2][1] = cy;

        auto r00 = aux.R[0][0];
        auto r01 = aux.R[1][0];
        auto r02 = aux.R[2][0];
        auto r10 = aux.R[0][1];
        auto r11 = aux.R[1][1];
        auto r12 = aux.R[2][1];
        auto r20 = aux.R[0][2];
        auto r21 = aux.R[1][2];
        auto r22 = aux.R[2][2];

        glm::mat4 T(1.0f);
        T[0][0] = r00;
        T[1][0] = r01;
        T[2][0] = r02;
        T[0][1] = r10;
        T[1][1] = r11;
        T[2][1] = r12;
        T[0][2] = r20;
        T[1][2] = r21;
        T[2][2] = r22;
        T[3][0] = m_channelInfo.calibration.aux.P[0][3] / m_channelInfo.calibration.aux.P[0][0]; // tx in mm
        T[3][1] = m_channelInfo.calibration.aux.P[1][3] / m_channelInfo.calibration.aux.P[0][0]; // ty in mm
        m_channelInfo.KColorMatExtrinsic = T;
        m_channelInfo.KColorMat = K;
    }

    void LibMultiSenseConnector::setup() {

    }

    uint8_t *LibMultiSenseConnector::getImage() {
        return nullptr;
    }


    /*

    bool LibMultiSenseConnector::getExposure(short m_channelID, bool hasAuxCamera) {
        std::scoped_lock<std::mutex> lock(m_channelMutex);
        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to get exposure on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to query exposure configuration");
            return false;
        }

        if (hasAuxCamera) {
            status = channelMap[channelID]->ptr()->getAuxImageConfig(infoMap[channelID].auxImgConf);
            if (crl::multisense::Status_Ok != status) {
                std::string errString = crl::multisense::Channel::statusString(status);
                Log::Logger::getInstance()->error("Unable to query getAuxImageConfig in getExposure. Error: {}",
                                                  errString);

                if (status == crl::multisense::Status_Unsupported) {
                    Log::Logger::getInstance()->error(
                            "Viewer thinks this camera has aux camera but LibMultiSense says it is unsupported");
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    void LibMultiSenseConnector::setIMUConfig(uint32_t tableIndex, crl::multisense::RemoteHeadChannel channelID) {
        if (!infoMap[channelID].hasIMUSensor) {
            Log::Logger::getInstance()->trace("No IMU Sensor present on device");
            return;
        }
        if (tableIndex > infoMap[channelID].imuMaxTableIndex) {
            Log::Logger::getInstance()->warning("Table index is higher than allowed {}. max allowed: {}", tableIndex,
                                                infoMap[channelID].imuMaxTableIndex);
            return;
        }

        auto conf = infoMap[channelID].imuSensorConfigs;
        for (auto &config: conf) {
            config.rateTableIndex = tableIndex;
        }

        // Set config
        auto channelP = channelMap[channelID]->ptr();

        crl::multisense::Status status = channelP->setImuConfig(false, 0, conf);

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Failed to set imu config: {}",
                                                crl::multisense::Channel::statusString(status));
            return;
        } else {
            Log::Logger::getInstance()->info("Set imu config table index {}", tableIndex);

        }
    }

  */

}