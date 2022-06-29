//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_CRLVIRTUALCAMERA_H
#define MULTISENSE_CRLVIRTUALCAMERA_H


#include "CRLBaseInterface.h"

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

    bool connect(const std::string& ip) override;
    void updateCameraInfo() override;
    void start(std::string string, std::string dataSourceStr) override;
    void stop(std::string dataSourceStr) override;


private:

    void update();
};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
