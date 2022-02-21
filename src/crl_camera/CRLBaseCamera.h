//
// Created by magnus on 2/20/22.
//

#ifndef MULTISENSE_CRLBASECAMERA_H
#define MULTISENSE_CRLBASECAMERA_H


#include <MultiSense/MultiSenseChannel.hh>
#include "MultiSense/src/model_loaders/MeshModel.h"

class CRLBaseCamera {

public:

    struct MeshData {
        void *vertices{};
        uint32_t vertexCount{};
        uint32_t *indices{};
        uint32_t indexCount{};

        Buffer mvp;
        Buffer pointCloud;

        MeshData() = default;

        MeshData(uint32_t vertexCount, uint32_t indexCount) {
            this->indexCount = indexCount;
            this->vertexCount = vertexCount;
            vertices = new MeshModel::Model::Vertex[vertexCount +
                                                    1];          // Here I add +1 for padding just to be safe for later
            indices = new uint32_t[indexCount + 1];              // Here I add +1 for padding just to be safe for later
        }

        ~MeshData() {
            free(vertices);
            delete[] indices;
        }

    };

    virtual void initialize() = 0;

    virtual void start() = 0;

    virtual void stop() = 0;

    virtual MeshData *getStream() = 0;


private:


};


#endif //MULTISENSE_CRLBASECAMERA_H
