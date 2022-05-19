//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <MultiSense/src/crl_camera/CRLBaseCamera.h>
#include "glm/glm.hpp"

class CRLPhysicalCamera : CRLBaseCamera {
public:

    bool online = false;
    bool play = false;
    bool modeChange = false;

    glm::mat4 kInverseMatrix{};
    crl::multisense::image::Header *stream = nullptr;

    std::vector<crl::multisense::DataSource> enabledSources;


    CRLPhysicalCamera() : CRLBaseCamera() {

    }


    void connect(const std::string& ip);
    void start(std::string string, std::string dataSourceStr) override;
    void update();
    void stop( std::string dataSourceStr) override;

    CameraInfo getInfo();
    PointCloudData *getStream() override;

    ~CRLPhysicalCamera();

    crl::multisense::image::Config getImageConfig() const;


    std::string dataSourceToString(unsigned int d);

    unsigned int stringToDataSource(const std::string &d);

    crl::multisense::image::Header getImage(unsigned int source);

    std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> getImage();

    std::unordered_set<crl::multisense::DataSource> supportedSources();

    static void setDelayedPropertyThreadFunc(void * context);

    void setup(uint32_t i, uint32_t i1);
};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
