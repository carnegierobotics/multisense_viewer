//
// Created by magnus on 3/1/22.
//

#include "CRLPhysicalCamera.h"


void CRLPhysicalCamera::connect() {



}

void CRLPhysicalCamera::start() {


}

void CRLPhysicalCamera::stop() {

}

void CRLPhysicalCamera::update(Base::Render render) {


}

CRLBaseCamera::PointCloudData *CRLPhysicalCamera::getStream() {
    return meshData;
}

CRLPhysicalCamera::~CRLPhysicalCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}

CRLBaseCamera::ImageData *CRLPhysicalCamera::getImageData() {

    return imageData;
}
