#ifndef LOGGER_H_
#define LOGGER_H_

// C++ Header File(s)
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#if __has_include(<source_location>)

#   include <source_location>

#   define HAS_SOURCE_LOCATION
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
            }info ;
            const VkRender::Device* dev = nullptr;
            std::unordered_map<crl::multisense::RemoteHeadChannel, std::unordered_map<std::string, uint32_t>> sourceReceiveMapCounter;
            std::vector<std::string> enabledSources;
            std::vector<std::string> requestedSources;
            std::vector<std::string> disabledSources;
            double upTime = 0.0f;
            bool ignoreMissingStatusUpdate = false;
        } device;
        /// SingleLayout Preview
        struct {
            CRLCameraDataType textureType = AR_CAMERA_IMAGE_NONE;
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
            glm::vec3 cameraFront;
        } camera;

    };

    class Logger {
    public:

        static Logger *getInstance() noexcept;

        static Metrics *getLogMetrics() noexcept;

        // Interface for Error Log
        void _error(const char *text) noexcept;

        /**@brief Using templates to allow user to use formattet logging.
     * @refitem @FormatString Is used to obtain m_Name of calling func, file and line number as default parameter */
        template<typename... Args>
        void error(const FormatString &format, Args &&... args) {
            vinfo(format, fmt::make_format_args(args...));

        }

        void error(const FormatString &format, fmt::format_args args) {
#if defined(HAS_SOURCE_LOCATION) || defined(HAS_SOURCE_LOCATION_EXPERIMENTAL)
            const auto &loc = format.m_Loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.m_Str, args);

            std::string preText = fmt::format("{}:{}: ", loc.file_name(), loc.line());
            preText.append(s);
            std::size_t found = preText.find_last_of('/');
            std::string msg = preText.substr(found + 1);

            _error(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), m_Format.m_Str, args);
            _error(s.c_str());

#endif

        }

        // Interface for Alarm Log
        void alarm(const char *text) noexcept;

        void alarm(std::string &text) noexcept;

        void alarm(std::ostringstream &stream) noexcept;

        // Interface for Always Log
        void always(const char *text) noexcept;

        void always(std::string &text) noexcept;

        void always(std::ostringstream &stream) noexcept;

        // Interface for Buffer Log
        void buffer(const char *text) noexcept;

        void buffer(std::string &text) noexcept;

        void buffer(std::ostringstream &stream) noexcept;

        // Interface for Info Log

        //void info(std::ostringstream& stream) noexcept;


        //void info(std::string &text, const source::source_location &m_Loc = source::source_location::current()) noexcept;

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
            const char* separator = "/";
#endif
            std::size_t found = preText.find_last_of(separator);
            std::string msg = preText.substr(found + 1);
            msg.append(s);

            msg = msg.insert(0, (std::to_string(frameNumber) + "  "));
            _info(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), m_Format.m_Str, args);
            s = s.insert(0, (std::to_string(frameNumber) + "  ")) ;
            _info(s.c_str());
#endif

        }


        // Interface for Trace log
        void trace(const char *text) noexcept;

        void trace(std::string &text) noexcept;

        void trace(std::ostringstream &stream) noexcept;

        // Interface for Debug log
        void debug(const char *text) noexcept;

        void debug(std::string &text) noexcept;

        void debug(std::ostringstream &stream) noexcept;

        // Error and Alarm log must be always flashing
        // Hence, there is no interfce to control error and alarm logs

        // Interfaces to control log levels
        void updateLogLevel(LogLevel logLevel);

        void enaleLog();  // Enable all log levels
        void disableLog(); // Disable all log levels, except error and alarm

        // Interfaces to control log Types
        void updateLogType(LogType logType);

        void enableConsoleLogging();

        void enableFileLogging();

        uint32_t frameNumber = 0;

        void operator=(const Logger &obj) = delete;

    protected:
        Logger();

        ~Logger();

        // Wrapper function for lock/unlock
        // For Extensible feature, lock and unlock should be in protected
        void lock();

        void unlock();

        std::string getCurrentTime();

    private:
        /*
        void info(const char *fmt, ...);
         */
        void _info(const char *text) noexcept;

        void logIntoFile(std::string &data);

        void logOnConsole(std::string &data);


    private:
        static Logger *m_Instance;
        static Metrics *m_Metrics;
        std::ofstream m_File;

        std::mutex m_Mutex{};

        LogLevel m_LogLevel{};
        LogType m_LogType{};

    };

} // End of namespace

#endif // End of _LOGGER_H_

