//
// Created by magnus on 11/3/22.
//

#ifndef MULTISENSE_VIEWER_SHAREDDATA_H
#define MULTISENSE_VIEWER_SHAREDDATA_H

#include <cstring>
#include "string"

class SharedData {
public:

    explicit SharedData(size_t sharedMemorySize){
        data = malloc(sharedMemorySize);
    }
    ~SharedData(){
        free(data);
    }

    template<typename T>
    void put(T* t, size_t size = 1){
        std::memcpy(data, t, sizeof(t) * size);
    }

    std::string destination;
    std::string source;

    void* data = nullptr;


};
#endif //MULTISENSE_VIEWER_SHAREDDATA_H
