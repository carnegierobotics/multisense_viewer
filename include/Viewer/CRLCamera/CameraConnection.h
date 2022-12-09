/**
 * @file: MultiSense-Viewer/include/Viewer/CRLCamera/CameraConnection.h
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
 *   2022-3-21, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include <memory>
#include <simpleini/SimpleIni.h>

#include "Viewer/Tools/ThreadPool.h"
#include "Viewer/Core/Definitions.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"

#define MAX_FAILED_STATUS_ATTEMPTS 8

namespace VkRender::MultiSense {
	/**
	 * Class handles the bridge between the GUI interaction and actual communication to camera
	 * Also handles all configuration with local network adapter
	 */
	class CameraConnection {
	public:
		CameraConnection() = default;

		~CameraConnection();

		/**Pointer to actual camera object*/
		std::unique_ptr<CRLPhysicalCamera> camPtr;
		/**Pointer to thread-pool commonly used for UI blocking operations*/
		std::unique_ptr<VkRender::ThreadPool> pool;

		/**@brief Called once per frame with a handle to the devices UI information block
		 * @param devices vector of devices 1:1 relationship with elements shown in sidebar
		 * @param[in] shouldConfigNetwork if user have ticked the "configure network" checkbox
		 * @param[in] isRemoteHead if the connected m_Device is a remote head, also selected by user
		 */
        void onUIUpdate(std::vector<VkRender::Device> &devices, bool shouldConfigNetwork);

		/**@brief Writes the current state of *dev to crl.ini configuration file
		 * @param[in] dev which profile to save to crl.ini
		 */
		void saveProfileAndDisconnect(VkRender::Device* dev);

	private:
		/**@brief file m_Descriptor to configure network settings on Linux */
		int m_FD = -1;
		/** @brief get status attempt counter */
		int m_FailedGetStatusCount = 0;
		/** @brief get status timer */
		std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> queryStatusTimer;
		/**@brief mutex to prevent multiple threads to communicate with camera.
		 * could be omitted if threadpool will always consist of one thread */
		std::mutex writeParametersMtx{};
		std::mutex statusCountMutex{};

		/**
		 * @brief Function called once per update by \refitem onUIUpdate if we have an active m_Device
		 * @param[out] dev which profile this m_Device is connected to
		 */
		void updateActiveDevice(VkRender::Device* dev);

		/**@brief Update system network settings if requested by user or autocorrect is chosen
		 *
		 * @param[out] dev which profile is selected
		 * @param[in] b should configure network
		 * @return if the network adapter were successfully configured
		 */
		bool setNetworkAdapterParameters(VkRender::Device& dev, bool b);

		/**@brief Get profile from .ini file if the serial number is recognized.
		 * @param[in] dev Which profile to update
		 */
		void getProfileFromIni(VkRender::Device& dev) const;

		/**@brief Create a user readable list of the possible camera modes*/
		void
			initCameraModes(std::vector<std::string>* modes, std::vector<crl::multisense::system::DeviceMode> vector);

		// Add ini m_Entry with log lines
		/**@brief Add a .ini m_Entry and log it*/
		static void addIniEntry(CSimpleIniA* ini, std::string section, std::string key, std::string value);

		/**@brief Delete a .ini m_Entry and log it*/
		static void deleteIniEntry(CSimpleIniA* ini, std::string section, std::string key, std::string value);

		// Caller functions for every *Task function is meant to be a threaded function
		/**@brief static function given to the threadpool to configure exposure of the sensor.
		 * @param[in] context pointer to the calling context
		 * @param[in] arg1 pointer to exposure params block
		 * @param[in] dev Which profile to update
		 * @param[in] index Which remote-head to select
		 * */
		static void setExposureTask(void* context, ExposureParams* arg1, VkRender::Device* dev,
			crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief static function given to the threadpool to configure the white balance of the sensor.
		 * @param[in] context pointer to the calling context
		 * @param[in] arg1 pointer to WhiteBalanceParams params block
		 * @param[in] dev Which profile to update
		 * @param[in] index Which remote-head to select
		 * */
		static void setWhiteBalanceTask(void* context, WhiteBalanceParams* arg1, VkRender::Device* dev,
			crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief static function given to the threadpool to configure lighting of the sensor.
		 * @param[in] context pointer to the calling context
		 * @param[in] arg1 pointer to the Lighting params block
		 * @param[in] dev Which profile to update
		 * @param[in] index Which remote-head to select
		 * */
		static void setLightingTask(void* context, LightingParams* arg1, VkRender::Device* dev,
			crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief static function given to the threadpool to configure exposure of the sensor.
		 * @param[in] context pointer to the calling context
		 * @param[in] arg1 What resolution to choose
		 * @param[in] index Which remote-head to select
		 * */
		static void
			setResolutionTask(void* context, CRLCameraResolution arg1, VkRender::Device* dev,
				crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief Set parameters to the sensor. Grouped together as in the UI
		 * @param[in] context pointer to the callers context
		 * @param[in] fps framerate to request
		 * @param[in] gain gain value
		 * @param[in] gamma gamma value
		 * @param[in] spfs stereo post filter strength
		 * @param[in] hdr enable hdr?
		 * @param[in] dev Which profile to update
		 * @param[in] index Which remotehead to select
		 */
		static void setAdditionalParametersTask(void* context, float fps, float gain, float gamma, float spfs,
			bool hdr, VkRender::Device* dev,
			crl::multisense::RemoteHeadChannel index
		);

		/**@brief Task to connect a CRL camera
		 * @param[in] context pointer to the callers context
		 * @param[in] dev What profile to connect to
		 * @param[in] remoteHead boolean to connect to remote head
		 * @param[in] config boolean to determine if application should set network settings
		 */
		static void connectCRLCameraTask(void* context, VkRender::Device* dev, bool remoteHead, bool config);

		/**@brief Request to start a stream
		 * @param[in] context pointer to the callers context
		 * @param[in] src What source to request start
		 * @param[in] remoteHeadIndex id of remote head to select
		 */
		static void startStreamTask(void* context, std::string src,
			crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief Request to stop a stream
		 * @param[in] context pointer to the callers context
		 * @param[in] src What source to request stop
		 * @param[in] remoteHeadIndex id of remote head to select
		 */
		static void stopStreamTask(void* context, std::string src,
			crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief Request to stop a stream
		 * @param[in] context pointer to the callers context
		 * @param[in] remoteHeadIndex id of remote head to select
		 * @param[out] msg if a status was received. This object is filled with the latest information
		 */
		static void getStatusTask(void* context, crl::multisense::RemoteHeadChannel remoteHeadIndex);

		/**@brief Update the UI block using the active information block from the physical camera
		 * @param[in] dev profile to update UI from
		 * @param[in] remoteHeadIndex id of remote head
		 */
		void
			updateFromCameraParameters(VkRender::Device* dev, crl::multisense::RemoteHeadChannel remoteHeadIndex) const;

		/**@brief Filter the unsupported sources defined by \ref maskArrayAll*/
		void filterAvailableSources(std::vector<std::string>* sources, std::vector<uint32_t> maskVec, crl::multisense::RemoteHeadChannel idx);

        /**
         * @brief task to set the extrinsic/intrinsic calibration using yml files
		 * @param[in] context pointer to the callers context
         * @param[in] intrinsicFilePath path of intrisics.yml
         * @param[in] extrinsicFilePath path of extrinsics.yml
		 * @param[in] remoteHeadIndex id of remote head to select
		 * @param[out] success returns if the calibration was successfully set. Used to update the UI element.
         */
        static void setCalibrationTask(void *context, const std::string & intrinsicFilePath, const std::string & extrinsicFilePath, crl::multisense::RemoteHeadChannel index, bool* success);

        /**
         * @brief task to set the extrinsic/intrinsic calibration using yml files
		 * @param[in] context pointer to the callers context
         * @param saveLocation directory to save the calibration files
		 * @param[in] remoteHeadIndex id of remote head to select
		 * @param[out] success returns if the calibration was successfully set. Used to update the UI element.
         */
        static void getCalibrationTask(void *context, const std::string& saveLocation, crl::multisense::RemoteHeadChannel index, bool* success);


        /**@brief MaskArray to sort out unsupported streaming modes. Unsupported for this application*/
		std::vector<uint32_t> maskArrayAll = {
				crl::multisense::Source_Luma_Left,
				crl::multisense::Source_Luma_Rectified_Left,
				crl::multisense::Source_Disparity_Left,
				crl::multisense::Source_Luma_Right,
				crl::multisense::Source_Luma_Rectified_Right,
				crl::multisense::Source_Chroma_Rectified_Aux,
				crl::multisense::Source_Luma_Aux,
				crl::multisense::Source_Luma_Rectified_Aux,
				crl::multisense::Source_Chroma_Aux,

		};

		std::vector<uint32_t> maskArrayUnused{
								crl::multisense::Source_Compressed_Aux,
				crl::multisense::Source_Compressed_Rectified_Aux,
									crl::multisense::Source_Compressed_Right,
				crl::multisense::Source_Compressed_Rectified_Right,
									crl::multisense::Source_Compressed_Left,
				crl::multisense::Source_Compressed_Rectified_Left,
		};

		void updateUIDataBlock(VkRender::Device& dev);

    };

}
#endif //MULTISENSE_CAMERACONNECTION_H
