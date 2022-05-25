//
// Created by magnus on 15/05/21.
//

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H

#include "glm/glm.hpp"

namespace Basil{
    struct VideoTexture {
        std::vector<unsigned char*> pixels{};
        uint64_t imageSize{};
        uint32_t width{};
        uint32_t height{};
    };


    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv0;
        glm::vec2 uv1;
        glm::vec4 joint0;
        glm::vec4 weight0;
    };

    struct Primitive {
        uint32_t firstIndex{};
        uint32_t indexCount{};
        uint32_t vertexCount{};
        bool hasIndices{};
        //Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount);
        //void setBoundingBox(glm::vec3 min, glm::vec3 max);
    };
}

typedef enum CRLCameraDataType {
    CrlPointCloud,
    CrlGrayscaleImage,
    CrlColorImageYUV420,
    CrlNone, CrlDisparityImage, CrlImage
} CRLCameraDataType;

typedef enum CRLCameraType {
    DEFAULT_CAMERA_IP,
    CUSTOM_CAMERA_IP,
    VIRTUAL_CAMERA
} CRLCameraType;


#endif //MULTISENSE_DEFINITIONS_H
