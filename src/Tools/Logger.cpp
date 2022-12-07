
// C++ Header File(s)
#include <iostream>
#include <ctime>
#include <vector>

#ifdef WIN32
#define semPost(x) SetEvent(x)
#define semWait(x, y) WaitForSingleObject(x, y)
#else
#include<bits/stdc++.h>
#define semPost(x) sem_post(x)
#endif

#include "Viewer/Tools/Logger.h"


// Code Specific Header Files(s)
using namespace std;
namespace Log {


    Logger *Logger::m_Instance = nullptr;
    VkRender::ThreadPool *Logger::m_ThreadPool = nullptr;
    Metrics* Logger::m_Metrics = nullptr;

// Log file m_Name. File m_Name should be change from here only
    const string logFileName = "logger.log";

    Logger::Logger() {
        m_File.open(logFileName.c_str(), ios::out | ios::app);
        m_LogLevel = LOG_LEVEL_TRACE;
        m_LogType = FILE_LOG;

    }


    Logger::~Logger() {
        m_File.close();
        delete m_Instance;
        delete m_Metrics;
        delete m_ThreadPool;
    }

    Logger *Logger::getInstance() noexcept {
        if (m_Instance == nullptr) {
            m_Instance = new Logger();
            m_Metrics = new Metrics();
            m_ThreadPool = new VkRender::ThreadPool(1);
        }
        return m_Instance;
    }

     Metrics * Logger::getLogMetrics() noexcept {
        return m_Metrics;
    }

    void Logger::logIntoFile(void* ctx, std::string &data) {
        auto * app = static_cast<Logger*> (ctx);
        app->m_Mutex.lock();
        app->m_File << getCurrentTime() << "  " << data << endl;
        app->m_Metrics->logQueue.push(data);
        app->m_Mutex.unlock();
    }

    void Logger::logOnConsole(std::string &data) {
        cout << getCurrentTime() << "  " << data << endl;
    }

    string Logger::getCurrentTime() {
        string currTime;
        //Current date/time based on current time
        time_t now = time(0);
        // Convert current time to string
        currTime.assign(ctime(&now));

        // Last charactor of currentTime is "\n", so remove it
        string currentTime = currTime.substr(0, currTime.size() - 1);
        return currentTime;
    }

// Interface for Error Log
    void Logger::error(const char *text) noexcept {
        string data;
        data.append("[ERROR]: ");
        data.append(text);

        // ERROR must be capture
        if (m_LogType == FILE_LOG) {
            m_ThreadPool->Push(Logger::logIntoFile, this, data);
        } else if (m_LogType == CONSOLE) {
            m_ThreadPool->Push(Logger::logOnConsole, data);
        }
    }


// Interface for Info Log
    void Logger::infoInternal(const char *text) noexcept {
        string data;
        data.append("[INFO]: ");
        data.append(text);

        if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_INFO)) {
            m_ThreadPool->Push(Logger::logIntoFile, this, data);
        } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_INFO)) {
            m_ThreadPool->Push(Logger::logOnConsole, data);
        }

    }

// Interface for Always Log
    void Logger::always(std::string text) noexcept {
        string data;
        data.append("[ALWAYS]: ");
        data.append(text);

        // No check for ALWAYS logs
        if (m_LogType == FILE_LOG) {
            m_ThreadPool->Push(Logger::logIntoFile, this, data);
        } else if (m_LogType == CONSOLE) {
            m_ThreadPool->Push(Logger::logOnConsole, data);
        }
    }

};