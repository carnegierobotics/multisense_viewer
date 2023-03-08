/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/Logger.h
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
#ifndef LOGGER_H_
#define LOGGER_H_

// C++ Header File(s)
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "Viewer/Tools/ThreadPool.h"

#if __has_include(<source_location>)

#include <source_location>

#define HAS_SOURCE_LOCATION
#elif __has_include(<experimental/filesystem>)
#   include <experimental/source_location>
#   define HAS_SOURCE_LOCATION_EXPERIMENTAL
#else
#   define NO_SOURCE_LOCATION
#endif

#include <fmt/core.h>
#include <mutex>

#ifdef WIN32
#include <process.h>
#else
// POSIX Socket Header File(s)
#include <cerrno>
#include <pthread.h>

#endif

#include <queue>
#include <unordered_map>
#include "Viewer/Core/Definitions.h"

namespace Log {
    // Direct Interface for logging into log file or console using MACRO(s)
#define LOG_ERROR(x)    Logger::getInstance()->error(x)
#define LOG_ALARM(x)       Logger::getInstance()->alarm(x)
#define LOG_ALWAYS(x)    Logger::getInstance()->always(x)
#define LOG_INFO(x)     Logger::getInstance()->info(x)
#define LOG_BUFFER(x)   Logger::getInstance()->buffer(x)
#define LOG_TRACE(x)    Logger::getInstance()->trace(x)
#define LOG_DEBUG(x)    Logger::getInstance()->debug(x)

    // enum for LOG_LEVEL
    typedef enum LOG_LEVEL {
        DISABLE_LOG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_BUFFER = 3,
        LOG_LEVEL_TRACE = 4,
        LOG_LEVEL_DEBUG = 5,
        ENABLE_LOG = 6,
    } LogLevel;

    // enum for LOG_TYPE
    typedef enum LOG_TYPE {
        NO_LOG = 1,
        CONSOLE = 2,
        FILE_LOG = 3,
    } LogType;

    struct FormatString {
        fmt::string_view m_Str;
#ifdef HAS_SOURCE_LOCATION
        std::source_location m_Loc;

        FormatString(const char *str, const std::source_location &loc = std::source_location::current()) : m_Str(str),
                                                                                                           m_Loc(loc) {}

#endif

#ifdef HAS_SOURCE_LOCATION_EXPERIMENTAL
        std::experimental::source_location m_Loc;
        FormatString(const char *m_Str, const  std::experimental::source_location &m_Loc =  std::experimental::source_location::current()) : m_Str(m_Str), m_Loc(m_Loc) {}
#endif

#ifdef  NO_SOURCE_LOCATION
        FormatString(const char *m_Str) : m_Str(m_Str) {}
#endif
    };

    struct Metrics {
        // InfoLogs
        std::queue<std::string> logQueue;
        /// MultiSense device
        struct {
            struct {
                std::string apiBuildDate;
                uint32_t apiVersion;
                std::string firmwareBuildDate;
                uint32_t firmwareVersion;
                uint64_t hardwareVersion;
                uint64_t hardwareMagic;
                uint64_t sensorFpgaDna;
            } info;
            const VkRender::Device *dev = nullptr;
            std::unordered_map<crl::multisense::RemoteHeadChannel, std::unordered_map<std::string, uint32_t>> sourceReceiveMapCounter;
            std::unordered_map<crl::multisense::RemoteHeadChannel, uint32_t> imuReceiveMapCounter;

            std::vector<std::string> enabledSources;
            std::vector<std::string> requestedSources;
            std::vector<std::string> disabledSources;
            double upTime = 0.0f;
            bool ignoreMissingStatusUpdate = false;
        } device;
        /// SingleLayout Preview
        struct {
            CRLCameraDataType textureType = CRL_CAMERA_IMAGE_NONE;
            uint32_t width = 0, height = 0;
            uint32_t texWidth = 0, texHeight = 0;
            CRLCameraResolution res = CRL_RESOLUTION_NONE;
            std::string src;
            bool usingDefaultTexture = false;
            int empty = 0;
        } preview;

        struct {
            float yaw = 0;
            float pitch = 0;
            glm::vec3 pos;
            glm::vec3 rot;
            glm::vec3 cameraFront;
        } camera;

    };

    class Logger {
    public:

        static Logger *getInstance() noexcept;

        static Metrics *getLogMetrics() noexcept;

        void errorInternal(const char *text) noexcept;

        /**@brief Using templates to allow user to use formattet logging.
     * @refitem @FormatString Is used to obtain m_Name of calling func, file and line number as default parameter */
        template<typename... Args>
        void error(const FormatString &format, Args &&... args) {
            verror(format, fmt::make_format_args(args...));
        }

        void verror(const FormatString &format, fmt::format_args args) {
#if defined(HAS_SOURCE_LOCATION) || defined(HAS_SOURCE_LOCATION_EXPERIMENTAL)
            const auto &loc = format.m_Loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.m_Str, args);
            std::string preText = fmt::format("{}:{}: ", loc.file_name(), loc.line());
            preText.append(s);
            std::size_t found = preText.find_last_of('/');
            std::string msg = preText.substr(found + 1);
            errorInternal(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), m_Format.m_Str, args);
            _error(s.c_str());

#endif
        }

        void warningInternal(const char *text) noexcept;

        /**@brief Using templates to allow user to use formattet logging.
* @refitem @FormatString Is used to obtain m_Name of calling func, file and line number as default parameter */
        template<typename... Args>
        void warning(const FormatString &format, Args &&... args) {
            vwarning(format, fmt::make_format_args(args...));
        }

        void vwarning(const FormatString &format, fmt::format_args args) {
#if defined(HAS_SOURCE_LOCATION) || defined(HAS_SOURCE_LOCATION_EXPERIMENTAL)
            const auto &loc = format.m_Loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.m_Str, args);
            std::string preText = fmt::format("{}:{}: ", loc.file_name(), loc.line());
            preText.append(s);
            std::size_t found = preText.find_last_of('/');
            std::string msg = preText.substr(found + 1);
            warningInternal(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), m_Format.m_Str, args);
            _error(s.c_str());

#endif
        }

        /**@brief Using templates to allow user to use formattet logging.
         * @refitem @FormatString Is used to obtain m_Name of calling func, file and line number as default parameter */
        template<typename... Args>
        void info(const FormatString &format, Args &&... args) {
            vinfo(format, fmt::make_format_args(args...));
        }
        void vinfo(const FormatString &format, fmt::format_args args) {
#if defined(HAS_SOURCE_LOCATION) || defined(HAS_SOURCE_LOCATION_EXPERIMENTAL)
            const auto &loc = format.m_Loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.m_Str, args);

            std::string preText = fmt::format(" {}:{}: ", loc.file_name(), loc.line());
#ifdef WIN32
            const char * separator = "\\";
#else
            const char *separator = "/";
#endif
            std::size_t found = preText.find_last_of(separator);
            std::string msg = preText.substr(found + 1);
            msg.append(s);
            msg = msg.insert(0, (std::to_string(frameNumber) + "  "));
            infoInternal(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), m_Format.m_Str, args);
            s = s.insert(0, (std::to_string(frameNumber) + "  ")) ;
            infoInternal(s.c_str());
#endif
        }
        uint32_t frameNumber = 0;
        void operator=(const Logger &obj) = delete;
        void always(std::string text) noexcept;

    protected:
        Logger();

        ~Logger();

        static std::string getCurrentTime();

    private:
        void infoInternal(const char *text) noexcept;

        static void logOnConsole(std::string &data);

        static void logIntoFile(void* ctx, std::string &data);

    private:
        static Logger *m_Instance;
        static VkRender::ThreadPool* m_ThreadPool;
        static Metrics *m_Metrics;
        std::ofstream m_File;

        std::mutex m_Mutex{};

        LogLevel m_LogLevel{};
        LogType m_LogType{};

    };

} // End of namespace

#endif // End of _LOGGER_H_

