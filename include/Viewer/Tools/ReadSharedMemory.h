/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/ReadSharedMemory.h
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
 *   2022-11-29, mgjerde@carnegierobotics.com, Created file.
 **/

#include <iostream>
#include <cstdio>
#include <cstring>
#include <Viewer/Tools/Json.hpp>

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
    std::vector<VkRender::EntryConnectDevice> entries;
    size_t logLine = 0;
    nlohmann::json jsonObj;

public:
    int fd = -1;
    bool stopRequested = false;
    bool isOpen = false;
    std::chrono::steady_clock::time_point time;
    std::string autoConnectVersion;

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
        if (isOpen){
            munmap(memPtr, ByteSize);
            close(fd);
            sem_close(semPtr);
            unlink(BackingFile);
        }
    }

    std::vector<VkRender::EntryConnectDevice> getResult() {
        return entries;
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
                    //std::ofstream o("pretty.json");
                    //o << std::setw(4) << jsonObj << std::endl;

                    if (jsonObj.contains("Command")){
                        if (jsonObj["Command"] == "Stop"){
                            stopRequested = true;
                            return true;
                        }
                    }

                    if (jsonObj.contains("Version") && autoConnectVersion.empty()) {
                        autoConnectVersion = jsonObj["Version"];
                        Log::Logger::getInstance()->info("Using AutoConnect version: {}", autoConnectVersion);
                    }

                    if (jsonObj.contains("Result")) {
                        auto res = jsonObj["Result"];
                        VkRender::EntryConnectDevice entry{};
                        for(size_t i = 0; i < res.size(); i++) {

                            entry.interfaceName = res[i]["Name"];
                            entry.interfaceIndex = res[i]["Index"];
                            entry.IP = res[i]["AddressList"][0];
                            entry.cameraName = res[i]["CameraNameList"][0];

                            bool addNewEntry = true;
                            for (const auto &taken: entries) {
                                if (taken.interfaceName == entry.interfaceName && taken.IP == entry.IP) {
                                    addNewEntry = false;
                                }
                            }
                            if (addNewEntry)
                                entries.emplace_back(entry);

                            // can be null on linux
                            if (res.contains("Description")) {
                                entry.description = res[i]["Description"];
                            }
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
    nlohmann::json send;
    void sendStopSignal() {
        if (!isOpen)
            return;

        send["Command"] = "Stop";

        if (semPtr == (void *) -1)
            logError("sem_open");
        strcpy(memPtr + (ByteSize / 2), to_string(send).c_str());
        if (sem_post(semPtr) < 0)
            logError("sem_post");
    }
    void setIpConfig(int index) {
        if (!isOpen || index < 0)
            return;

        send["SetIP"] = "SetIP";
        send["index"] = std::to_string(index);


        if (semPtr == (void *) -1)
            logError("sem_open");
        strcpy(memPtr + (ByteSize / 2), to_string(send).c_str());
        if (sem_post(semPtr) < 0)
            logError("sem_post");
    }

    void logError(const char *msg) {
        Log::Logger::getInstance()->error("{}: strerror: {}", msg, strerror(errno));
    }
};

#else
#define SharedBufferSize 65536

class ReaderWindows {

    std::vector<VkRender::EntryConnectDevice> entries;
    size_t logLine = 0;
    nlohmann::json jsonObj{};

    HANDLE hMapFile{};
    char *pBuf = nullptr;

public:
    bool stopRequested = false;
    bool isOpen = false;
    std::chrono::steady_clock::time_point time;
    std::string autoConnectVersion;

    ReaderWindows() {
        time = std::chrono::steady_clock::now();
    }

    void open(bool byPassTimer = false) {
        // Only try once a second
        if (((std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - time).count() < 1) || isOpen) && !byPassTimer) {
            return;
        }
        time = std::chrono::steady_clock::now();

        TCHAR szName[] = TEXT("Global\\MyFileMappingObject");

        hMapFile = OpenFileMapping(
                FILE_MAP_WRITE,   // read/write access
                FALSE,                 // do not inherit the name
                szName);               // name of mapping object

        if (hMapFile == nullptr) {
            logError("Could not create file mapping object...");
            return;
        }

        pBuf = reinterpret_cast<char *> (MapViewOfFile(hMapFile,   // handle to map object
                                                       FILE_MAP_WRITE, // read/write permission
                                                       0,
                                                       0,
                                                       SharedBufferSize));
        if (pBuf == nullptr) {
            logError("Could not map view of file...");

            CloseHandle(hMapFile);
            return;
        }
        Log::Logger::getInstance()->trace("Opened shared memory handle");
        isOpen = true;
    }

    ~ReaderWindows() {
        if (isOpen) {
            UnmapViewOfFile(pBuf);
            CloseHandle(hMapFile);
        }
    }

    std::vector<VkRender::EntryConnectDevice> getResult() {
        return entries;
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
        float oneSecond = 1.0f;
        if ((std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - time).count() > oneSecond) && isOpen) {

            time = std::chrono::steady_clock::now();
            /* use semaphore as a mutex (lock) by waiting for writer to increment it */
            std::string str(pBuf);

            Log::Logger::getInstance()->trace("Reading from shared memory (AutoConnect). Message size: {}", str.size());
            if (!str.empty()) {
                jsonObj = nlohmann::json::parse(str);

                // write prettified JSON to another file
                //std::ofstream o("pretty.json");
                //o << std::setw(4) << jsonObj << std::endl;

                if (jsonObj.contains("Command")) {
                    if (jsonObj["Command"] == "Stop") {
                        stopRequested = true;
                        Log::Logger::getInstance()->info("AutConnect has request to stop");
                        return true;
                    }
                }

                if (jsonObj.contains("Version") && autoConnectVersion.empty()) {
                    autoConnectVersion = jsonObj["Version"];
                    Log::Logger::getInstance()->info("Using AutoConnect version: {}", autoConnectVersion);
                }

                if (jsonObj.contains("Result")) {
                    Log::Logger::getInstance()->info("AutConnect message contains results: count = {}",
                                                     jsonObj["Result"].size());
                    auto res = jsonObj["Result"];
                    VkRender::EntryConnectDevice entry{};
                    for (size_t i = 0; i < res.size(); i++) {

                        entry.interfaceName = res[i]["Name"];
                        entry.interfaceIndex = res[i]["Index"];
                        entry.IP = res[i]["AddressList"][0];
                        entry.cameraName = res[i]["CameraNameList"][0];

                        bool addNewEntry = true;
                        for (const auto &taken: entries) {
                            if (taken.interfaceName == entry.interfaceName && taken.IP == entry.IP) {
                                addNewEntry = false;
                            }
                        }
                        if (addNewEntry)
                            entries.emplace_back(entry);

                        // can be null on linux
                        if (res.contains("Description")) {
                            entry.description = res[i]["Description"];
                        }
                    }
                }

                memset(pBuf, 0x00, SharedBufferSize / 2);
                Log::Logger::getInstance()->trace("Clearing shared memory, ready for next message");
                if (!str.empty())
                    return true;
            }
        }
        return false;
    }

    nlohmann::json send;
    void sendStopSignal() {
        if (!isOpen)
            return;

        send["Command"] = "Stop";

        strcpy_s(pBuf + (SharedBufferSize / 2), (SharedBufferSize / 2), nlohmann::to_string(send).c_str());
        Log::Logger::getInstance()->info("Sent stop signal to AutoConnect");
    }
    void setIpConfig(int index) {
        if (!isOpen || index < 0)
            return;

        send["SetIP"] = "SetIP";
        send["index"] = std::to_string(index);


        strcpy_s(pBuf + (SharedBufferSize / 2), (SharedBufferSize / 2), nlohmann::to_string(send).c_str());
        Log::Logger::getInstance()->info("Sent stop signal to AutoConnect");
    }


    void logError(const std::string &msg) {
        Log::Logger::getInstance()->error("{}: GetLastError: {}", msg.c_str(), GetLastError());

    }
};

#endif