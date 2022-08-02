//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_CRLVIRTUALCAMERA_H
#define MULTISENSE_CRLVIRTUALCAMERA_H


#include <semaphore.h>
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

    bool connect(const std::string &ip) override;

    void updateCameraInfo() override;

    void preparePointCloud(uint32_t width, uint32_t height) override;

    void start(std::string string, std::string dataSourceStr) override;

    void stop(std::string dataSourceStr) override;

    void getCameraStream(std::string stringSrc, crl::multisense::image::Header **src,
                         crl::multisense::image::Header **src2 = nullptr) override;



private:

    void update();


    int width = 0, height = 0;
    AVFrame videoFrame[5];
    int bufferSize = 0;

    int r1, items = 0;
    sem_t notEmpty, notFull;
    bool runDecodeThread = false;
    pthread_t producer;
    int frameIndex = 0;


    int childProcessDecode();
    static void* decode(void* arg);


};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
