//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLMULTISENSES30_H
#define MULTISENSE_CRLMULTISENSES30_H

#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLMultiSenseS30 : CRLBaseCamera {
public:
    std::string description;
    std::string data;
    PointCloudData* meshData;
    int point = 0;

    void initialize() override;
    void start() override;
    void update(Base::Render render);
    void stop() override;
    PointCloudData *getStream() override;

    ~CRLMultiSenseS30();

private:

};


#endif //MULTISENSE_CRLMULTISENSES30_H
