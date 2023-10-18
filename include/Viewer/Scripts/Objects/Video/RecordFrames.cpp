/**
 * @file: MultiSense-Viewer/src/Scripts/Objects/Video/RecordFrames.cpp
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
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/

#include <filesystem>
#include <tiffio.h>
#include <RosbagWriter/RosbagWriter.h>

#include "Viewer/Scripts/Objects/Video/RecordFrames.h"
#include "Viewer/Scripts/Private/RecordUtilities.h"

void RecordFrames::setup() {
}

void RecordFrames::update() {

    bool shouldCreateThreadPool = saveImage || savePointCloud || saveIMUData;

    if (shouldCreateThreadPool && threadPool == nullptr) {
        Log::Logger::getInstance()->trace("Creating RecordFrames ThreadPool");
        threadPool = std::make_unique<VkRender::ThreadPool>(3);
    }
    if (saveImage)
        saveImageToFile();
    if (savePointCloud)
        savePointCloudToFile();
    if (saveIMUData)
        saveIMUDataToFile();

}

void RecordFrames::saveImageToFile() {
    // For each enabled source in all windows
    for (auto &src: sources) {
        if (src == "Idle")
            continue;

        // Check if we saved a lot more images of this type than others. So skip it
        bool skipThisFrame = false;
        auto count = savedImageSourceCount.find(src);
        std::string otherSrc;
        for (const auto &i: savedImageSourceCount) {
            if (src == i.first)
                continue;
            if (count->second > i.second + 3) {
                skipThisFrame = true;
                otherSrc = i.first;
                break;
            }
        }
        if (skipThisFrame) {
            Log::Logger::getInstance()->trace(
                    "Skipping saving frame {} of source {} since we have to many compared to {} which has {}",
                    savedImageSourceCount[src] + 1, src, otherSrc, savedImageSourceCount[otherSrc]);
            continue;
        }

        {
            const auto &conf = renderData.crlCamera->getCameraInfo(0).imgConf;
            auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(),
                                                               conf.height(), false, true);
            if (renderData.crlCamera->getCameraStream(src, tex.get(), 0)) {
                if (src == "Color Aux" || src == "Color Rectified Aux") {
                    if (tex->m_Id != tex->m_Id2)
                        continue;
                }
                if (threadPool->getTaskListSize() < MAX_IMAGES_IN_QUEUE) {
                    // Dont overwrite already saved images
                    if (lastSavedImagesID[src] == tex->m_Id)
                        continue;
                    Log::Logger::getInstance()->trace("Pushing task: saveImageToFile. ID: {}, Comp: {}. Count: {}",
                                                      tex->m_Id, fileFormat.c_str(), saveImageCount[src]);
                    threadPool->Push(saveImageToFileAsync,
                                     Utils::CRLSourceToTextureType(src),
                                     saveFolderImage,
                                     src,
                                     tex,
                                     fileFormat, writer, std::ref(rosbagWriterMutex)
                    );
                    lastSavedImagesID[src] = tex->m_Id;
                    saveImageCount[src]++;
                    savedImageSourceCount[src]++;
                } else if (threadPool->getTaskListSize() >= MAX_IMAGES_IN_QUEUE &&
                           (fileFormat == "tiff" || fileFormat == "rosbag")) {
                    Log::Logger::getInstance()->warning("Record image queue is full. Dropping frame: {}, source: {}",
                                                        tex->m_Id, src);
                }
            }
        }
    }
}


void RecordFrames::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    for (VkRender::Device &dev: uiHandle->devices) {


        if (dev.state != CRL_STATE_ACTIVE)
            continue;

        sources.clear();
        for (const auto &window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource)) {
                sources.emplace_back(window.second.selectedSource);

                // Check if "foo" exists in the map
                if (savedImageSourceCount.find(window.second.selectedSource) == savedImageSourceCount.end() &&
                    window.second.selectedSource != "Idle") {
                    Log::Logger::getInstance()->trace("Added source {} to saved image counter in record frames",
                                                      window.second.selectedSource);
                    savedImageSourceCount[window.second.selectedSource] = 0;

                    saveImageCount[window.second.selectedSource] = 0;
                }
            }
        }
        if (hashVector(sources) != hashVector(prevSources)) {
            savedImageSourceCount.clear();
        }
        prevSources = sources;
        saveFolderImage = dev.record.frameSaveFolder;
        saveFolderPointCloud = dev.record.pointCloudSaveFolder;
        saveFolderIMUData = dev.record.imuSaveFolder;
        saveImage = dev.record.frame;
        fileFormat = dev.saveImageCompressionMethod;
        isRemoteHead = dev.isRemoteHead;
        savePointCloud = dev.record.pointCloud;
        saveIMUData = dev.record.imu;
        useAuxColor = dev.useAuxForPointCloudColor == 1;
        // IF we started a saveImage process
        if (saveImage && !prevSaveState) {
            writer = std::make_shared<CRLRosWriter::RosbagWriter>();
        }
        // If we stopped a saveImage process
        if (!saveImage && prevSaveState) {

            {
                // Shared pointer gymnastics. Make sure a thread is the last remaining user of the rosbagwriter so we dont block the UI
                threadPool->Push(finishWritingRosbag,
                                 writer, std::ref(rosbagWriterMutex)
                );
                writer.reset(); // This must happen from a thread}
            }


            auto &json = dev.record.metadata.JSON;

            std::string activeSources;
            for (const auto &[key, value]: saveImageCount) {
                activeSources += key + ",";
                json["data_info"][key]["number_of_images"] = std::to_string(value);

                std::string comp = fileFormat;
                if (fileFormat == "tiff" && key.find("Color") != std::string::npos) {
                    json["data_info"][key]["fileFormat"] = "ppm";
                    comp = "ppm";
                } else {
                    json["data_info"][key]["fileFormat"] = dev.saveImageCompressionMethod;
                }
                json["paths"][key] =
                        saveFolderImage + std::string("/") + std::string(key) + std::string("/") + comp + "/";
                json["paths"]["timestamp_file"][key] =
                        saveFolderImage + std::string("/") + std::string(key) + std::string("/") + "timestamps.csv";
            }

            // Find the latest occurrence of ","
            size_t pos = activeSources.rfind(',');
            // Check if "," was found
            if (pos != std::string::npos) {
                // Erase the ","
                activeSources.erase(pos, 1);
            }

            json["data_info"]["sources"] = activeSources;
            json["data_info"]["frames_per_second"] = dev.parameters.stereo.fps;

            // Get todays date:
            auto t = std::time(nullptr);
            std::tm tm;

#if defined(WIN32) // MSVC compiler
            localtime_s(&tm, &t);
#elif defined(__GNUC__) || defined(__clang__) // GCC or Clang compiler
            localtime_r(&t, &tm);
#else
            auto tmp = std::localtime(&t);
            if (tmp) tm = *tmp; // Fallback for other compilers, but be aware it's not thread-safe
#endif

            std::ostringstream oss;
            oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");

            json["data_info"]["session_end"] = oss.str();


            std::filesystem::path saveFolder = dev.record.frameSaveFolder;
            saveFolder.append("metadata.json");
            std::ofstream file(saveFolder);
            file << json.dump(4);  // 4 spaces as indentation
            saveImageCount.clear();

        }

        prevSaveState = saveImage;
    }
}

void RecordFrames::saveIMUDataToFile() {
    std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> gyro;
    std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> accel;

    if (renderData.crlCamera->getIMUData(0, &gyro, &accel)) {
        // If we successfully fetch both image streams into a texture
        // Calculate pointcloud then save to file
        if (threadPool->getTaskListSize() < MAX_IMAGES_IN_QUEUE) {
            threadPool->Push(saveIMUDataToFileAsync, saveFolderIMUData, gyro, accel);
        } else {
            Log::Logger::getInstance()->trace("Skipping saving imu data to file. Too many tasks in taskList {}",
                                              threadPool->getTaskListSize());
        }
    }
}

void RecordFrames::savePointCloudToFile() {
    const auto &conf = renderData.crlCamera->getCameraInfo(0).imgConf;
    auto disparityTex = std::make_shared<VkRender::TextureData>(CRL_DISPARITY_IMAGE, conf.width(), conf.height(),
                                                                false, true);

    std::shared_ptr<VkRender::TextureData> colorTex;
    CRLCameraDataType texType = CRL_GRAYSCALE_IMAGE;
    std::string source = "Luma Rectified Left";
    if (useAuxColor) {
        source = "Color Rectified Aux";
        texType = CRL_COLOR_IMAGE_YUV420;
    }
    colorTex = std::make_shared<VkRender::TextureData>(texType, conf.width(), conf.height(), false, true);

    // If we successfully fetch both image streams into a texture
    if (renderData.crlCamera->getCameraStream(source, colorTex.get(), 0) &&
        renderData.crlCamera->getCameraStream("Disparity Left", disparityTex.get(), 0)) {
        // Calculate pointcloud then save to file
        if (threadPool->getTaskListSize() < MAX_IMAGES_IN_QUEUE) {
            threadPool->Push(savePointCloudToPlyFile, saveFolderPointCloud, disparityTex,
                             colorTex, useAuxColor,
                             renderData.crlCamera->getCameraInfo(0).QMat,
                             renderData.crlCamera->getCameraInfo(0).pointCloudScale,
                             renderData.crlCamera->getCameraInfo(0).focalLength
            );
        }
    }
}

// Function to calculate the hash of a vector of strings
size_t RecordFrames::hashVector(const std::vector<std::string> &v) {
    size_t seed = 0;
    for (const auto &str: v) {
        std::hash<std::string> hasher;
        seed ^= hasher(str) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}


void RecordFrames::saveIMUDataToFileAsync(const std::filesystem::path &saveDirectory,
                                          const std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> &gyro,
                                          const std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> &accel) {

    Log::Logger::getInstance()->trace("Saving accel and gyro data to file at: {}", saveDirectory.string());
    // Create folders if no exists
    if (!std::filesystem::is_directory(saveDirectory) ||
        !std::filesystem::exists(saveDirectory)) { // Check if src folder exists
        std::filesystem::create_directories(saveDirectory); // create src folder
    }

    std::filesystem::path accelLocation = saveDirectory;
    accelLocation /= "accel.txt";
    std::filesystem::path gyroLocation = saveDirectory;
    gyroLocation /= "gyro.txt";

    std::ofstream accelFile(accelLocation, std::ios::app);
    if (accelFile.is_open()) { // Check if the file was successfully opened
        for (const auto &a: accel) {
            accelFile << std::fixed << std::setprecision(6) << a.time << " "
                      << a.x << " " << a.y << " " << a.z << std::endl;
        }
    }
    std::ofstream gyroFile(gyroLocation, std::ios::app);
    if (gyroFile.is_open()) { // Check if the file was successfully opened
        for (const auto &g: gyro) {
            gyroFile << std::fixed << std::setprecision(6) << g.time << " "
                     << g.x << " " << g.y << " " << g.z << std::endl;
        }
    }

}

void RecordFrames::savePointCloudToPlyFile(const std::filesystem::path &saveDirectory,
                                           std::shared_ptr<VkRender::TextureData> &depthTex,
                                           std::shared_ptr<VkRender::TextureData> &colorTex, bool useAuxColor,
                                           const glm::mat4 &Q, const float &scale, const float &focalLength) {

    std::filesystem::path fileLocation = saveDirectory;
    fileLocation /= (std::to_string(depthTex->m_Id) + ".ply");
    // Dont overwrite already saved images
    if (std::filesystem::exists(fileLocation))
        return;
    // Create folders if no exists
    if (!std::filesystem::is_directory(saveDirectory) ||
        !std::filesystem::exists(saveDirectory)) { // Check if src folder exists
        std::filesystem::create_directories(saveDirectory); // create src folder
    }

    struct WorldPoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float r = 0.0f, g = 0.0f, b = 0.0f;

        WorldPoint(float xx, float yy, float zz, float rr, float gg, float bb) : x(xx), y(yy), z(zz), r(rr), g(gg),
                                                                                 b(bb) {
        }

        WorldPoint() = default;
    };

    const auto *disparityP = reinterpret_cast<const uint16_t *>(depthTex->data);
    const auto *leftRectifiedP = reinterpret_cast<const uint8_t *>(colorTex->data);
    uint32_t width = depthTex->m_Width;
    uint32_t height = depthTex->m_Height;
    uint32_t minDisparity = 5;

    std::vector<WorldPoint> points;
    points.resize(height * width);

    std::vector<uint8_t> colorRGB(colorTex->m_Width * colorTex->m_Height * 3);
    if (useAuxColor) {
        RecordUtility::ycbcrToRGB(colorTex->data, colorTex->data2, colorTex->m_Width, colorTex->m_Height,
                                  colorRGB.data());
    }

    size_t index = 0;
    float d = 0;
    float invB = 0;
    size_t colorIndex = 0;
    glm::vec4 imgCoords(0.0f), worldCoords(0.0f);
    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            index = h * width + w;
            // MultiSense 16 bit disparity images are stored in 1/16 of a pixel.
            d = static_cast<float>(disparityP[index]) / 16.0f;
            if (d < minDisparity) {
                continue;
            }
            invB = focalLength / d;
            imgCoords = glm::vec4(w, h, d, 1);
            worldCoords = Q * imgCoords * (1 / invB);
            worldCoords = worldCoords / worldCoords.w * scale;

            if (useAuxColor) {
                colorIndex = (h * width + w) * 3;
                points[index] = WorldPoint(worldCoords.x,
                                           worldCoords.y,
                                           worldCoords.z,
                                           static_cast<float>(colorRGB[colorIndex]),
                                           static_cast<float>(colorRGB[colorIndex + 1]),
                                           static_cast<float>(colorRGB[colorIndex + 2]));
            } else {
                points[index] = WorldPoint(worldCoords.x,
                                           worldCoords.y,
                                           worldCoords.z,
                                           static_cast<float>(leftRectifiedP[index]),
                                           static_cast<float>(leftRectifiedP[index]),
                                           static_cast<float>(leftRectifiedP[index]));
            }
        }
    }
    std::stringstream ss;
    ss << "ply\n";
    ss << "format ascii 1.0\n";
    ss << "element vertex " << points.size() << "\n";
    ss << "property float x\n";
    ss << "property float y\n";
    ss << "property float z\n";
    ss << "property uchar red\n";
    ss << "property uchar green\n";
    ss << "property uchar blue\n";
    ss << "end_header\n";
    for (const auto &point: points) {
        ss << point.x << " " << point.y << " " << point.z << " " << point.r << " " << point.g << " " << point.b << "\n";
    }
    std::ofstream ply(fileLocation.string());
    ply << ss.str();
}

void RecordFrames::saveImageToFileAsync(CRLCameraDataType type, const std::string &path, std::string &stringSrc,
                                        std::shared_ptr<VkRender::TextureData> &ptr,
                                        std::string &fileFormat,
                                        std::shared_ptr<CRLRosWriter::RosbagWriter> rosbagWriter,
                                        std::mutex &rosbagWriterMut) {
    // Create folders. One for each Source
    std::filesystem::path saveDirectory{};
    std::replace(stringSrc.begin(), stringSrc.end(), ' ', '_');
    if (type == CRL_COLOR_IMAGE_YUV420 && fileFormat == "tiff") {
        fileFormat = "ppm";
    }
    saveDirectory = path + "/" + stringSrc + "/" + fileFormat + "/";
    std::filesystem::path fileLocation = saveDirectory.string() + std::to_string(ptr->m_Id) + "." + fileFormat;
    // Create folders if no exists
    if (fileFormat != "rosbag") {
        if (!std::filesystem::is_directory(saveDirectory) ||
            !std::filesystem::exists(saveDirectory)) { // Check if src folder exists
            std::filesystem::create_directories(saveDirectory); // create src folder
        }
    }

    // Create file
    if (fileFormat == "tiff" || fileFormat == "ppm") {
        switch (type) {
            case CRL_POINT_CLOUD:
                break;
            case CRL_GRAYSCALE_IMAGE:
                RecordUtility::writeTIFFImage(fileLocation, ptr, 8);
                break;
            case CRL_DISPARITY_IMAGE:
                RecordUtility::writeTIFFImage(fileLocation, ptr, 16);
                break;
            case CRL_COLOR_IMAGE_YUV420:
                RecordUtility::writePPMImage(fileLocation, ptr);
                break;
            case CRL_CAMERA_IMAGE_NONE:
            case CRL_COLOR_IMAGE_RGBA:
                break;
            default:
                Log::Logger::getInstance()->info("Recording not specified for this format");
                break;
        }
    } else if (fileFormat == "png") {
        switch (type) {
            case CRL_POINT_CLOUD:
                break;
            case CRL_GRAYSCALE_IMAGE:
                RecordUtility::writePNGImageGrayscale(fileLocation, ptr);
                break;
            case CRL_DISPARITY_IMAGE:
                RecordUtility::writePNGImageGrayscale(fileLocation, ptr, true);
                break;
            case CRL_COLOR_IMAGE_YUV420:
                RecordUtility::writePNGImageColor(fileLocation, ptr);
                break;
            default:
                Log::Logger::getInstance()->info("Recording not specified for this format");
                break;
        }
    } else if (fileFormat == "rosbag") {
        // Topic is stream type
        std::string topic = "/" + stringSrc;
        // Replacing all spaces with underscores
        for (size_t pos = topic.find(" "); pos != std::string::npos; pos = topic.find(" ", pos)) {
            topic.replace(pos, 1, "_");
        }
        // Synchronize writing to bag.
        std::scoped_lock<std::mutex> lock(rosbagWriterMut);
        // Open bag if not opened
        rosbagWriter->open(std::filesystem::path(path) / "MultiSense.bag");
        // Add connection if not added
        auto conn = rosbagWriter->getConnection(topic, "sensor_msgs/Image");
        int64_t timestamp;
        timestamp = (static_cast<int64_t>(ptr->m_TimeSeconds) * 1000000000LL)
                    + (static_cast<int64_t>(ptr->m_TimeMicroSeconds) * 1000LL);

        std::string encoding = CRLSourceToRosImageEncodingString(type);
        std::vector<uint8_t> data;
        uint32_t step = ptr->m_Width;
        uint32_t imageSize = ptr->m_Width * ptr->m_Height;
        // convert color to rgb
        switch (type) {
            case CRL_GRAYSCALE_IMAGE:
                data = rosbagWriter->serializeImage(ptr->m_Id, timestamp, ptr->m_Width, ptr->m_Height, ptr->data,
                                                    imageSize,
                                                    encoding, step);
                break;
            case CRL_DISPARITY_IMAGE:
                data = rosbagWriter->serializeImage(ptr->m_Id, timestamp, ptr->m_Width, ptr->m_Height, ptr->data,
                                                    imageSize * 2,
                                                    encoding, step * 2);
                break;
            case CRL_COLOR_IMAGE_YUV420: {
                std::vector<uint8_t> output(imageSize * 3);
                RecordUtility::ycbcrToRGB(ptr->data, ptr->data2, ptr->m_Width, ptr->m_Height, output.data());
                data = rosbagWriter->serializeImage(ptr->m_Id, timestamp, ptr->m_Width, ptr->m_Height, output.data(),
                                                    imageSize * 3,
                                                    encoding, step * 3);
            }
                break;
            default:
                Log::Logger::getInstance()->warning("Format not supported for Rosbag");
                break;
        }
        Log::Logger::getInstance()->trace("Got rosbag connection {}, topic: {}, type: {}, encoding {}, size {}",
                                          conn.id, conn.topic, conn.msgtype, encoding, data.size());

        rosbagWriter->write(conn, timestamp, data);
    }
    // Write to the timestamp file
    if (fileFormat != "rosbag")
        RecordUtility::writeTimestamps(fileLocation, ptr);
}

void RecordFrames::finishWritingRosbag(std::shared_ptr<CRLRosWriter::RosbagWriter> rosbagWriter,
                                       std::mutex &rosbagWriterMut) {
    std::scoped_lock<std::mutex> lock(rosbagWriterMut);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    Log::Logger::getInstance()->info("Releasing rosbagWriter");

}

