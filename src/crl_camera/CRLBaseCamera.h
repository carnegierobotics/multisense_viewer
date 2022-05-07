//
// Created by magnus on 2/20/22.
//

#ifndef MULTISENSE_CRLBASECAMERA_H
#define MULTISENSE_CRLBASECAMERA_H


#include <MultiSense/MultiSenseChannel.hh>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

typedef enum CRLCameraDataType {
    CrlPointCloud,
    CrlGrayscaleImage,
    CrlColorImageYUV420,
    CrlNone, CrlDisparityImage, CrlImage
} CRLCameraDataType;

typedef enum CRLCameraType {
    DEFAULT_CAMERA_IP,
    CUSTOM_CAMERA_IP,
    VIRTUAL_CAMERA
} CRLCameraType;

struct Image
{
    uint16_t height{0}, width{0};
    uint32_t size{0};
    int64_t frame_id{0};
    crl::multisense::DataSource source{};
    const void *data{nullptr};
};

struct BufferPair
{
    std::mutex swap_lock;
    crl::multisense::image::Header active, inactive;
    void *activeCBBuf{nullptr}, *inactiveCBBuf{nullptr};  // weird multisense BufferStream object, only for freeing reserved data later
    Image user_handle;

    void refresh()   // swap to latest if possible
    {
        std::scoped_lock lock(swap_lock);

        auto handleFromHeader = [](const auto &h) {
            return Image{static_cast<uint16_t>(h.height), static_cast<uint16_t>(h.width), h.imageLength, h.frameId, h.source, h.imageDataP};
        };

        if ((activeCBBuf == nullptr && inactiveCBBuf != nullptr) || // special case: first init
            (active.frameId < inactive.frameId))
        {
            std::swap(active, inactive);
            std::swap(activeCBBuf, inactiveCBBuf);
            user_handle = handleFromHeader(active);
        }
    }
};

class CRLBaseCamera {

public:
    CRLBaseCamera() = default;

    static constexpr uint16_t DEFAULT_WIDTH = 1920, DEFAULT_HEIGHT = 1080;

    std::unordered_map<crl::multisense::DataSource,BufferPair> buffers_;
    std::unordered_map<crl::multisense::DataSource,crl::multisense::image::Header> imagePointers;

    struct CameraInfo {
        crl::multisense::system::DeviceInfo devInfo;
        crl::multisense::image::Config imgConf;
        crl::multisense::system::NetworkConfig netConfig;
        crl::multisense::system::VersionInfo versionInfo;
        crl::multisense::image::Calibration camCal{};
        std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes;
        crl::multisense::DataSource supportedSources{0};
        std::vector<uint8_t*> rawImages;
    }cameraInfo;

    void prepare();
    bool connect(CRLCameraType type); // true if succeeds

    struct PointCloudData {
        void *vertices{};
        uint32_t vertexCount{};
        uint32_t *indices{};
        uint32_t indexCount{};

        /*
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
*/
        ~PointCloudData() {
            free(vertices);
            delete[] indices;
        }

    };

    PointCloudData *meshData{};

    virtual void initialize() {};

    virtual void start(std::string string, std::string dataSourceStr) = 0;

    virtual void stop(std::string dataSourceStr) = 0;

    virtual PointCloudData *getStream() = 0;

    bool connected = false;

    crl::multisense::Channel * cameraInterface{};

protected:
    void getCameraMetaData();

    void getVirtualCameraMetaData();

    void addCallbacks();

    static void imageCallback(const crl::multisense::image::Header &header, void *userDataP);

    void streamCallback(const crl::multisense::image::Header &image);

    void selectFramerate(float FPS);

    void selectDisparities(uint32_t disparities);

    void selectResolution(uint32_t RequestedWidth, uint32_t RequestedHeight);

    std::unordered_set<crl::multisense::DataSource> supportedSources();
};




#endif //MULTISENSE_CRLBASECAMERA_H
