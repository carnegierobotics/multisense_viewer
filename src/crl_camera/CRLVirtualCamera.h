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
#endif

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
    int width = 0, height = 0;

    bool connect(const std::string &ip) override;

    void updateCameraInfo() override;

    void preparePointCloud(uint32_t width, uint32_t height) override;

    void start(std::string string, std::string dataSourceStr) override;

    void stop(std::string dataSourceStr) override;

    CameraInfo getCameraInfo() override;

    void getCameraStream(crl::multisense::image::Header *stream) override;
    bool getCameraStream(ArEngine::MP4Frame* frame) override;
    void getCameraStream(std::string src, crl::multisense::image::Header *stream, crl::multisense::image::Header **src2) override;


private:

    void update();

    CameraInfo info{};
    AVFrame* videoFrame;
    int bufferSize = 0;
    std::string videoName = "None";
    bool decoded = false;

    int r1, items = 0;
#ifdef WIN32 // TODO USE MACROS INSTEAD AND DEFINE MACROS DEPENDING ON PLATFORM
    HANDLE notEmpty;
    HANDLE notFull;
    //HANDLE producer;
    DWORD ThreadID;
    //static void decode(LPVOID);
    static void* decode(void* arg);
    std::thread* producer;

#else
    sem_t notEmpty, notFull;
    pthread_t producer;
    static void* decode(void* arg);

#endif    
    bool runDecodeThread = false;


    int frameIndex = 0;
    std::vector<uint32_t> frameIDs;
    bool pauseThread = false;
    uint32_t lastFrame = 0;
    uint32_t idx = 0;

    int childProcessDecode();


    void getVideoMetadata();

    void saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame);
};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
