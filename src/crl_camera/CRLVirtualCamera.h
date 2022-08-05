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
    int width = 0, height = 0;

    bool connect(const std::string &ip) override;

    void updateCameraInfo() override;

    void preparePointCloud(uint32_t width, uint32_t height) override;

    void start(std::string string, std::string dataSourceStr) override;

    void stop(std::string dataSourceStr) override;

    void getCameraStream(crl::multisense::image::Header *stream) override;
    bool getCameraStream(ArEngine::MP4Frame* frame) override;


private:

    void update();

    AVFrame videoFrame[5];
    int bufferSize = 0;
    std::string videoName = "None";
    bool decoded = false;

    int r1, items = 0;
#ifdef WIN32
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
    bool pauseThread = false;


    int childProcessDecode();


    void getVideoMetadata();

    void saveFrameYUV420P(AVFrame *pFrame, int width, int height, int iFrame);
};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
