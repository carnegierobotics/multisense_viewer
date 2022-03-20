//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLPhysicalCamera : CRLBaseCamera {
public:

    explicit CRLPhysicalCamera(CRLCameraDataType type) : CRLBaseCamera() {
        CRLBaseCamera::prepare();
    }

    std::string description;
    std::string data;
    bool online = false;
    bool play = false;
    int point = 0;
    bool modeChange = false;
    std::vector<crl::multisense::DataSource> enabledSources;



    void connect();
    void start(std::string string, std::string dataSourceStr) override;
    void update(Base::Render render, crl::multisense::image::Header *pHeader);
    void stop( std::string dataSourceStr) override;

    CameraInfo getInfo();
    PointCloudData *getStream() override;

    ~CRLPhysicalCamera();

    crl::multisense::image::Config getImageConfig() const;

    std::unordered_set<crl::multisense::DataSource> supportedSources();

    std::string dataSourceToString(unsigned int d);

    unsigned int stringToDataSource(const std::string &d);

    crl::multisense::image::Header getImage(unsigned int source);

    std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> getImage();
};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
