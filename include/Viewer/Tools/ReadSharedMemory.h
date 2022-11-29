//
// Created by magnus on 11/28/22.
//

#include <iostream>
#include <cstdio>
#include <cstring>
#include <Viewer/Tools/json.hpp>

#include "Viewer/Tools/Logger.h"

#define ByteSize 65536
#define BackingFile "/mem"
#define AccessPerms 0777
#define SemaphoreName "sem"

#ifdef __linux__
#include <cstdlib>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

class ReaderLinux {
    caddr_t memPtr{};
    sem_t *semPtr{};
    VkRender::EntryConnectDevice entry;
    size_t logLine = 0;
    nlohmann::json jsonObj;

public:
    int fd = -1;
    bool stopRequested = false;
    bool isOpen = false;
    std::chrono::steady_clock::time_point time;

    ReaderLinux() {
        time = std::chrono::steady_clock::now();

    }

    void open() {
        // Only try once a second
        if ((std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - time).count() < 1) || isOpen) {
            return;
        }
        time = std::chrono::steady_clock::now();

        fd = shm_open(BackingFile, O_RDWR, AccessPerms);  /* empty to begin */
        if (fd < 0) {
            logError("Can't get file descriptor...");
            return;
        }

        /* get a pointer to memory */
        memPtr = static_cast<caddr_t>(mmap(nullptr,       /* let system pick where to put segment */
                                           ByteSize,   /* how many bytes */
                                           PROT_READ | PROT_WRITE, /* access protections */
                                           MAP_SHARED, /* mapping visible to other processes */
                                           fd,         /* file descriptor */
                                           0));         /* offset: start at 1st byte */
        if ((caddr_t) -1 == memPtr) {
            logError("Can't access segment...");
            return;
        }

        /* create a semaphore for mutual exclusion */
        semPtr = sem_open(SemaphoreName, /* name */
                          O_CREAT,       /* create the semaphore */
                          AccessPerms,   /* protection perms */
                          0);            /* initial value */
        if (semPtr == (void *) -1 || semPtr == nullptr) {
            logError("sem_open");
            return;
        }

        isOpen = true;
        Log::Logger::getInstance()->info("Opened shared memory handle");
    }

    ~ReaderLinux() {
        /* cleanup */
        munmap(memPtr, ByteSize);
        close(fd);
        sem_close(semPtr);
        unlink(BackingFile);
    }

    VkRender::EntryConnectDevice getResult() {
        return entry;
    }

    std::string getLogLine() {

        try {
            std::string str = jsonObj["Log"].at(logLine);
            logLine++;
            return str;
        } catch (...) {
            // Empty catch dont care
        }

        return "";
    }

    bool read() {
        // Only try once a second
        if ((std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - time).count() > 0.1f) && isOpen) {

            time = std::chrono::steady_clock::now();
            /* use semaphore as a mutex (lock) by waiting for writer to increment it */
            if (!sem_wait(semPtr)) { /* wait until semaphore != 0 */
                std::string str(memPtr);
                if (!str.empty()) {
                    jsonObj = nlohmann::json::parse(str);

                    // write prettified JSON to another file
                    std::ofstream o("pretty.json");
                    o << std::setw(4) << jsonObj << std::endl;

                    if (jsonObj.contains("Command")){
                        if (jsonObj["Command"] == "Stop"){
                            stopRequested = true;
                            return true;
                        }
                    }

                    if (jsonObj.contains("Result")) {
                        // can be null on linux
                        auto res = jsonObj["Result"];

                        entry.interfaceName = res[0]["Name"];

                        // can be null on linux
                        entry.interfaceIndex = res[0]["Index"];

                        // can be null on linux
                        entry.IP = res[0]["AddressList"][0];

                        // can be null on linux
                        entry.cameraName = res[0]["CameraNameList"][0];

                        // can be null on linux
                        if (res.contains("Description")) {
                            entry.description = res[0]["Description"];
                        }
                    }
                }
                memset(memPtr, 0x00, ByteSize / 2);
                sem_post(semPtr);
                if (!str.empty())
                    return true;
            }
        }

        return false;
    }

    void sendStopSignal() {
        if (!isOpen)
            return;
        nlohmann::json send = {
                {"Command", "Stop"}
        };
        if (semPtr == (void *) -1)
            logError("sem_open");
        strcpy(memPtr + (ByteSize / 2), to_string(send).c_str());
        if (sem_post(semPtr) < 0)
            logError("sem_post");
    }

    void logError(const char *msg) {
        Log::Logger::getInstance()->error("%s: strerror: %s", msg, strerror(errno));

    }
};

#else

class ReaderWindows {

    VkRender::EntryConnectDevice entry;
    size_t logLine = 0;
    nlohmann::json jsonObj;

public:
    bool stopRequested = false;
    bool isOpen = false;
    std::chrono::steady_clock::time_point time;

    ReaderWindows() {
        time = std::chrono::steady_clock::now();
    }

    void open() {
        // Only try once a second
        if ((std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - time).count() < 1) || isOpen) {
            return;
        }
        time = std::chrono::steady_clock::now();
        //isOpen = true;
        Log::Logger::getInstance()->info("Opened shared memory handle");
    }

    ~ReaderWindows() {

    }

    VkRender::EntryConnectDevice getResult() {
        return entry;
    }

    std::string getLogLine() {

        try {
            std::string str = jsonObj["Log"].at(logLine);
            logLine++;
            return str;
        } catch (...) {
            // Empty catch dont care
        }

        return "";
    }

    bool read() {
        // Only try once a second
        if ((std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - time).count() > 0.1f) && isOpen) {
                return false;
        }

            time = std::chrono::steady_clock::now();

        return false;
    }

    void sendStopSignal() {
        if (!isOpen)
            return;
    }

    void logError(const char *msg) {
        Log::Logger::getInstance()->error("%s: strerror: %s", msg, strerror(errno));

    }
};

#endif