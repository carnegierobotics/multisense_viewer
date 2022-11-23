//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECT_H
#define AUTOCONNECT_AUTOCONNECT_H

#define MAX_CONNECTION_ATTEMPTS 2
#define TIMEOUT_INTERVAL_SECONDS 5

#include <string>
#include <vector>
#include <thread>

class AutoConnect {
public:
    enum FoundCameraOnIp {
        FOUND_CAMERA = 0,
        NO_CAMERA_RETRY = 1,
        NO_CAMERA = 2
    };


    struct Result {
        Result() = default;

        Result(const char *name,
               uint8_t supp) { // By default, we want to initialize an adapter result with a name and support status
            this->networkAdapter = name;
            this->supports = supp;
        }

        bool supports{}; // 0: for bad, 1: for good
        bool searched = false;
        std::string cameraIpv4Address;
        std::string description;
        std::string networkAdapter;
        std::string networkAdapterLongName;
        uint32_t index{};
    } result;

    bool m_LoopAdapters = true;
    bool m_ListenOnAdapter = true;
    bool m_ShouldProgramRun = true;
    time_t startTime{};
    std::vector<Result> m_IgnoreAdapters{};
    //virtual std::vector<AutoConnect::Result> findEthernetAdapters(bool b, bool skipIgnored) = 0;
    virtual void start() = 0;

    virtual void onFoundAdapters(std::vector<Result> vector, bool logEvent) = 0;

    virtual AutoConnect::FoundCameraOnIp onFoundIp(std::string string, Result adapter, int camera_fd) = 0;

    virtual void onFoundCamera() = 0;

    virtual void stopAutoConnect() = 0;

    virtual bool isRunning() = 0;

    virtual void setShouldProgramRun(bool exit) = 0;

protected:
    int connectAttemptCounter = 0;
    std::thread *m_TAutoConnect = nullptr;
    std::thread *m_TAdapterSearch = nullptr;
    bool interrupt = false;
};


#endif //AUTOCONNECT_AUTOCONNECT_H
