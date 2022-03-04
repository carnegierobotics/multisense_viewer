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
} CRLCameraDataType;


class CRLBaseCamera {

public:
    explicit CRLBaseCamera(CRLCameraDataType type) {
        this->requestedDataType = type;

    }
    std::string DEFAULT_CAMERA_IP = "10.66.171.21";
    static constexpr uint16_t DEFAULT_WIDTH = 1920, DEFAULT_HEIGHT = 1080;
    CRLCameraDataType requestedDataType;

    crl::multisense::Channel * cameraInterface{};
    std::unique_ptr<crl::multisense::Channel> camInterface;
    // TODO: Hide all multisense types/impl details in camera_stream pimpl

    using ImgConf = crl::multisense::image::Config;
    using NetConf = crl::multisense::system::NetworkConfig;
    using CamCal = crl::multisense::image::Calibration;
    using DeviceMode = crl::multisense::system::DeviceMode;
    using DataSource = crl::multisense::DataSource;
    using VersionInfo = crl::multisense::system::VersionInfo;

    crl::multisense::system::DeviceInfo devInfo;

    void prepare();
    void connect(std::string& hostname); // true if succeeds

    struct PointCloudData {
        void *vertices{};
        uint32_t vertexCount{};
        uint32_t *indices{};
        uint32_t indexCount{};

        PointCloudData(uint32_t width, uint32_t height) {
            vertexCount = width * height;
            // Virtual class can generate some mesh data here
            vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));

            uint32_t v = 0;
            auto *vP = (MeshModel::Model::Vertex *) vertices;
            for (uint32_t x = 0; x < width; ++x) {
                for (uint32_t z = 0; z < height; ++z) {
                    MeshModel::Model::Vertex vertex{};
                    vertex.pos = glm::vec3((float) x / 50, 0.0f, (float) z / 50);
                    vertex.uv0 = glm::vec2((float) x / (float) width, (float) z / (float) height);
                    vP[v] = vertex;
                    v++;
                }
            }
        };

        ~PointCloudData() {
            free(vertices);
            delete[] indices;
        }

    };

    struct ImageData {
        struct {
            void *vertices{};
            uint32_t vertexCount{};
            uint32_t *indices{};
            uint32_t indexCount{};
        } quad;

        /**@brief Generates a Quad with texture coordinates */
        ImageData() {
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

    PointCloudData *meshData{};
    ImageData *imageData{};
    // gets latest image, old image is invalidated on next getImage



    virtual void initialize() {};

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual PointCloudData *getStream() = 0;


private:


};




#endif //MULTISENSE_CRLBASECAMERA_H
