//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_CRLVIRTUALCAMERA_H
#define MULTISENSE_CRLVIRTUALCAMERA_H

#ifdef WIN32
#include <windows.h>
#include <thread>
#else
#include <semaphore.h>
#include <queue>

#endif

#define MAX_STREAMS 5
#include "CRLBaseInterface.h"

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

    bool connect(const std::string &ip) override;

    void updateCameraInfo() override;
    void updateCameraInfo(uint32_t index);

    void preparePointCloud(uint32_t width, uint32_t height) override;

    void start(std::string string, std::string dataSourceStr) override;

    void start(std::string src, StreamIndex parent) override;

    void stop(std::string dataSourceStr) override;
    void stop(StreamIndex parent) override;

    CameraInfo getCameraInfo() override;

    bool getCameraStream(ArEngine::MP4Frame* frame, StreamIndex parent) override;
    bool getCameraStream(ArEngine::YUVTexture *tex) override;


private:

    void update();

    CameraInfo info{};
    int key = 0;
    std::unordered_map<StreamIndex, uint32_t > parentKeyMap;
    struct DecodeThreadArgs {
        CRLVirtualCamera* ctx;
        uint32_t index;
    };
    DecodeThreadArgs args[MAX_STREAMS];

    struct DecodeContainer{
#ifdef WIN32 // TODO USE MACROS INSTEAD AND DEFINE MACROS DEPENDING ON PLATFORM
        HANDLE notEmpty;
        HANDLE notFull;
        //HANDLE producer;
        DWORD ThreadID;
        //static void decode(LPVOID);
        static void* decode(void* arg);
        std::thread* producer;

#else
        sem_t notEmpty{}, notFull{};
        pthread_t producer{};

        static void *decode(void* args);

#endif
        bool runDecodeThread = false;
        AVFrame *videoFrame{};
        int bufferSize = 0;
        std::string videoName = "None";
        int width = 0, height = 0;


        int frameIndex = 0;
        std::array<uint32_t, 20> frameIDs{0};


        std::array<bool, 20> occupiedSlot{};
        bool pauseThread = false;
        uint32_t lastFrame = 0;
        uint32_t idx = 0;
        uint32_t idxFrame = 0;

    };

    std::array<DecodeContainer, MAX_STREAMS> container;

    int childProcessDecode(uint32_t index);


    void getVideoMetadata(uint32_t i);

    void saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame);
};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
