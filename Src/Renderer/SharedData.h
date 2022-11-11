//
// Created by magnus on 11/3/22.
//

#ifndef MULTISENSE_VIEWER_SHAREDDATA_H
#define MULTISENSE_VIEWER_SHAREDDATA_H

#include "string"

class SharedData {
public:

    explicit SharedData(size_t sharedMemorySize){
        data = calloc(sharedMemorySize, 1);
    }
    ~SharedData(){
        free(data);
    }

    template<typename T>
    void put(T* t, size_t extraSize, size_t copies = 1){
        std::memcpy(data, t, extraSize + sizeof(t) * copies);
    }

    std::string destination;
    std::string source;

    void* data = nullptr;


};
#endif //MULTISENSE_VIEWER_SHAREDDATA_H
