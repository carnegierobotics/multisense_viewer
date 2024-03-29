/**
 * @file: MultiSense-Viewer/include/Viewer/Renderer/SharedData.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-11-3, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_SHAREDDATA_H
#define MULTISENSE_VIEWER_SHAREDDATA_H

#include <string>

namespace VkRender {
    struct TopLevelScriptData {
        struct {
            std::vector<VkBuffer> *computeBuffer = nullptr;
            std::vector<Texture2D> *textureComputeTarget = nullptr;
            std::vector<Texture3D> *textureComputeTarget3D = nullptr;
            bool valid = false;
            bool reset = false;
        } compute;
    };
}

class SharedData {
public:

    explicit SharedData(size_t sharedMemorySize) {
        data = calloc(sharedMemorySize, 1);
    }

    ~SharedData() {
        free(data);
    }

    template<typename T>
    void put(T *t, size_t extraSize, size_t copies = 1) {
        std::memcpy(data, t, extraSize + sizeof(t) * copies);
    }

    std::string destination;
    std::string source;

    void *data = nullptr;


};

#endif //MULTISENSE_VIEWER_SHAREDDATA_H
