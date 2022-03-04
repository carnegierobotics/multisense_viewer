//
// Created by magnus on 2/20/22.
//

#ifndef MULTISENSE_CRLBASECAMERA_H
#define MULTISENSE_CRLBASECAMERA_H


#include <MultiSense/MultiSenseChannel.hh>
#include "MultiSense/src/model_loaders/MeshModel.h"
typedef enum CRLCameraDataType {
    CrlPointCloud,
    CrlImage
}CRLCameraDataType;


class CRLBaseCamera {

public:
    // TODO: Hide all multisense types/impl details in camera_stream pimpl
    using ImgConf = crl::multisense::image::Config;
    using NetConf = crl::multisense::system::NetworkConfig;
    using DevInfo = crl::multisense::system::DeviceInfo;
    using CamCal = crl::multisense::image::Calibration;
    using DeviceMode = crl::multisense::system::DeviceMode;
    using DataSource = crl::multisense::DataSource;
    using VersionInfo = crl::multisense::system::VersionInfo;


    const ImgConf& imgConf() const;
    const DevInfo& devInfo() const;
    const NetConf& netConf() const;
    const CamCal&  camCal() const;
    const VersionInfo versionInfo() const;
    std::vector<DeviceMode> deviceModes() const;

    bool updateNetConf(const NetConf &c);
    bool updateImgConf(const ImgConf &c);

    bool startStream(DataSource stream);
    bool stopStream(DataSource stream);

    bool connect(std::string hostname); // true if succeeds
    bool connected() const;

    struct PointCloudData {
        void *vertices{};
        uint32_t vertexCount{};
        uint32_t *indices{};
        uint32_t indexCount{};

        PointCloudData() = default;

        PointCloudData(uint32_t vertexCount, uint32_t indexCount) {
            this->indexCount = indexCount;
            this->vertexCount = vertexCount;
            vertices = new MeshModel::Model::Vertex[vertexCount +
                                                    1];          // Here I add +1 for padding just to be safe for later
            indices = new uint32_t[indexCount + 1];              // Here I add +1 for padding just to be safe for later
        }

        ~PointCloudData() {
            free(vertices);
            delete[] indices;
        }

    };
    struct ImageData {
        struct {
            void* vertices{};
            uint32_t vertexCount{};
            uint32_t *indices{};
            uint32_t indexCount{};
        }quad;

        /**@brief Generates a Quad with texture coordinates */
        ImageData(){
            int vertexCount = 4;
            int indexCount = 2 * 3;
            quad.vertexCount = vertexCount;
            quad.indexCount = indexCount;
            // Virtual class can generate some mesh data here
            quad.vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));
            quad.indices = static_cast<uint32_t *>(calloc(indexCount, sizeof(uint32_t)));

            auto *vP = (MeshModel::Model::Vertex *) quad.vertices;
            auto *iP = (uint32_t *) quad.indices;

            MeshModel::Model::Vertex vertex[4];
            vertex[0].pos = glm::vec3(0.0f, 0.0f, 0.0f);
            vertex[1].pos = glm::vec3(1.0f, 0.0f, 0.0f);
            vertex[2].pos = glm::vec3(0.0f, 0.0f, 1.0f);
            vertex[3].pos = glm::vec3(1.0f, 0.0f, 1.0f);

            vertex[0].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[1].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[2].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[3].normal = glm::vec3(0.0f, 1.0f, 0.0f);

            vertex[0].uv0 = glm::vec2(0.0f, 0.0f);
            vertex[1].uv0 = glm::vec2(1.0f, 0.0f);
            vertex[2].uv0 = glm::vec2(0.0f, 1.0f);
            vertex[3].uv0 = glm::vec2(1.0f, 1.0f);
            vP[0] = vertex[0];
            vP[1] = vertex[1];
            vP[2] = vertex[2];
            vP[3] = vertex[3];
            // indices
            iP[0] = 0;
            iP[1] = 1;
            iP[2] = 2;
            iP[3] = 1;
            iP[4] = 2;
            iP[5] = 3;
        }
    };
    // gets latest image, old image is invalidated on next getImage
    std::optional<PointCloudData> getImage(DataSource d);  // in a more general version, this would be a weak_ptr

    static constexpr char DEFAULT_CAMERA_IP[] = "10.66.171.21";
    static constexpr uint16_t DEFAULT_WIDTH = 1920, DEFAULT_HEIGHT = 1080;


    virtual void initialize() { };

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual PointCloudData *getStream() = 0;


private:


};


#endif //MULTISENSE_CRLBASECAMERA_H
