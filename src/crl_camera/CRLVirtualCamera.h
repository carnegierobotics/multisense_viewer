//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_CRLVIRTUALCAMERA_H
#define MULTISENSE_CRLVIRTUALCAMERA_H

#ifdef WIN32
#include <windows.h>
#define semaphore HANDLE
#else
#define semaphore sem_t
#include <semaphore.h>
#endif

#include <thread>
#include <array>

#define MAX_STREAMS 6
#include <MultiSense/src/crl_camera/CRLBaseInterface.h>

extern "C" {
#include <libavutil/frame.h>
};

class CRLVirtualCamera : public CRLBaseInterface {
public:
    explicit CRLVirtualCamera() : CRLBaseInterface() {
    }

    ~CRLVirtualCamera() override {
        // TODO FREE RESOURCES AS THIS CLASS IS REUSED
    }

    std::string description;
    std::string data;
    int point = 0;
    glm::mat4 kInverseMatrix;

    bool connect(const std::string& ip) override;

    void updateCameraInfo() override;
    void updateCameraInfo(uint32_t index);

    void preparePointCloud(uint32_t width, uint32_t height) override;

    bool start(CRLCameraResolution resolution, std::string dataSourceStr) override;

    void start(std::string src, StreamIndex parent) override;

    bool stop(std::string dataSourceStr) override;
    void stop(StreamIndex parent) override;

    CameraInfo getCameraInfo() override;

    bool getCameraStream(ArEngine::MP4Frame* frame, StreamIndex parent) override;
    bool getCameraStream(ArEngine::YUVTexture* tex) override;



    void update();

    CameraInfo info{};
    int key = 0;
    std::unordered_map<StreamIndex, uint32_t > parentKeyMap;
    struct DecodeThreadArgs {
        CRLVirtualCamera* ctx;
        uint32_t index;
    };
    DecodeThreadArgs args[MAX_STREAMS];

    struct DecodeContainer {
        //static void decode(LPVOID);
        std::thread* producer = nullptr;

        semaphore notEmpty{}, notFull{};
        static void* decode(void* arg);

        bool runDecodeThread = false;
        AVFrame* videoFrame{};
        int bufferSize = 0;
        std::string videoName = "None";
        int width = 0, height = 0;


        int frameIndex = 0;
        std::array<uint32_t, 20> frameIDs{ 0 };


        std::array<bool, 20> occupiedSlot{};
        bool pauseThread = false;
        uint32_t lastFrame = 0;
        uint32_t idx = 0;
        uint32_t idxFrame = 0;

    };

    std::array<DecodeContainer, MAX_STREAMS> container;

    int childProcessDecode(uint32_t index);


    bool getVideoMetadata(uint32_t i);

    void saveFrameYUV420P(AVFrame* pFrame, int width, int height, int iFrame);
};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
