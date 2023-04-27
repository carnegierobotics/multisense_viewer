/**
 * @file: MultiSense-Viewer/src/Tools/Logger.cpp
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
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/

// C++ Header File(s)
#include <iostream>
#include <ctime>
#include <vector>
#include <regex>


#ifdef WIN32
#define semPost(x) SetEvent(x)
#define semWait(x, y) WaitForSingleObject(x, y)
#define CTIME(time, size, timer) ctime_s(time, size, timer)
#else
#include<bits/stdc++.h>
#define semPost(x) sem_post(x)
#define CTIME(time, timer) ctime(timer)
#endif

#include "Viewer/Tools/Logger.h"

// Code Specific Header Files(s)
using namespace std;
namespace Log {


    Logger *Logger::m_Instance = nullptr;
    VkRender::ThreadPool *Logger::m_ThreadPool = nullptr;
    Metrics* Logger::m_Metrics = nullptr;
    Logger::Logger(const std::string& logFileName) {
        m_File.open(logFileName.c_str(), ios::out | ios::app);

        m_LogLevel = LOG_LEVEL_INFO;
        m_LogType = FILE_LOG;
    }


    Logger::~Logger() {
        m_File.close();
        delete m_Instance;
        delete m_Metrics;
        delete m_ThreadPool;
    }

    Logger *Logger::getInstance(const std::string& fileName) noexcept {
        if (m_Instance == nullptr) {
            m_Instance = new Logger(fileName);
            m_Metrics = new Metrics();
            m_ThreadPool = new VkRender::ThreadPool(1);
            m_Instance->info("Initialized logger instance, fileName: {}", fileName);
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
        //Current date/time based on current time
        // Convert current time to string
        std::time_t currentTime = std::time(nullptr);
        char timeString[26];

        #ifdef _WIN32
                ctime_s(timeString, sizeof(timeString), &currentTime);
        #else
                std::strcpy(timeString, std::ctime(&currentTime));
        #endif

        // Last character of currentTime is "\n", so remove it
        string currentTimeStr(timeString);
        return currentTimeStr.substr(0, currentTimeStr.size() - 1);
    }

// Interface for Error Log
    void Logger::errorInternal(const char *text) noexcept {
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

    void Logger::warningInternal(const char *text) noexcept {
        string data;
        data.append("[WARNING]: ");
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
    void Logger::traceInternal(const char *text) noexcept {
        string data;
        data.append("[TRACE]: ");
        data.append(text);

        if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
            m_ThreadPool->Push(Logger::logIntoFile, this, data);
        } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
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

    std::string Logger::filterFilePath(const std::string& input) {
        // Filter away the absolute file path given by std::source_location, both for anonymous and readable logs purpose.
        // Magic regex expression, courtesy of chatgpt4
        std::regex folderPathRegex(R"(^((?:[a-zA-Z]:[\\\/])?(?:[\w\s-]+[\\\/])+))");
        std::string result = std::regex_replace(input, folderPathRegex, "");


        return result;

    }

    void Logger::setLogLevel(LogLevel logLevel) {
        getInstance()->info("Setting log level to: {}", getLogStringFromEnum(logLevel));
        m_LogLevel = logLevel;
    }

};