///////////////////////////////////////////////////////////////////////////////
// @File Name:     Logger.h                                                  //
// @Author:        Pankaj Choudhary                                          //
// @Version:       0.0.1                                                     //
// @L.M.D:         13th April 2015                                           //
// @Description:   For Logging into file                                     //
//                                                                           // 
// Detail Description:                                                       //
// Implemented complete logging mechanism, Supporting multiple logging type  //
// like as file based logging, console base logging etc. It also supported   //
// for different log type.                                                   //
//                                                                           //
// Thread Safe logging mechanism. Compatible with VC++ (Windows platform)   //
// as well as G++ (Linux platform)                                           //
//                                                                           //
// Supported Log Type: ERROR, ALARM, ALWAYS, INFO, BUFFER, TRACE, DEBUG      //
//                                                                           //
// No control for ERROR, ALRAM and ALWAYS messages. These type of messages   //
// should be always captured.                                                //
//                                                                           //
// BUFFER log type should be use while logging raw buffer or raw messages    //
//                                                                           //
// Having direct interface as well as C++ Singleton inface. can use          //
// whatever interface want.                                                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

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



#if __has_include(<format>)
#include <format>
#endif


#include <fmt/core.h>
#include <mutex>

#ifdef WIN32
// Win Socket Header File(s)
#include <Windows.h>
#include <process.h>
#else
// POSIX Socket Header File(s)
#include <cerrno>
#include <pthread.h>

#endif

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
        fmt::string_view str;
#ifdef HAS_SOURCE_LOCATION
        std::source_location loc;
        FormatString(const char *str, const std::source_location &loc = std::source_location::current()) : str(str), loc(loc) {}
#endif

#ifdef HAS_SOURCE_LOCATION_EXPERIMENTAL
        std::experimental::source_location loc;
        FormatString(const char *str, const  std::experimental::source_location &loc =  std::experimental::source_location::current()) : str(str), loc(loc) {}
#endif

#ifdef  NO_SOURCE_LOCATION
        FormatString(const char *str) : str(str) {}
#endif



    };

    class Logger {
    public:
        static Logger *getInstance() noexcept;
        // Interface for Error Log
        void _error(const char *text) throw();

        /**@brief Using templates to allow user to use formattet logging.
     * @refitem @FormatString Is used to obtain name of calling func, file and line number as default parameter */
        template<typename... Args>
        void error(const FormatString &format, Args &&... args) {
            vinfo(format, fmt::make_format_args(args...));

        }

        void error(const FormatString &format, fmt::format_args args) {
#if defined(HAS_SOURCE_LOCATION) || defined(HAS_SOURCE_LOCATION_EXPERIMENTAL)
            const auto &loc = format.loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.str, args);

            std::string preText = fmt::format("{}:{}: ", loc.file_name(), loc.line());
            preText.append(s);
            std::size_t found = preText.find_last_of('/');
            std::string msg = preText.substr(found + 1);

            _error(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.str, args);
            _error(s.c_str());

#endif

        }

        // Interface for Alarm Log
        void alarm(const char *text) throw();

        void alarm(std::string &text) throw();

        void alarm(std::ostringstream &stream) throw();

        // Interface for Always Log
        void always(const char *text) throw();

        void always(std::string &text) throw();

        void always(std::ostringstream &stream) throw();

        // Interface for Buffer Log
        void buffer(const char *text) throw();

        void buffer(std::string &text) throw();

        void buffer(std::ostringstream &stream) throw();

        // Interface for Info Log

        //void info(std::ostringstream& stream) throw();


        //void info(std::string &text, const source::source_location &loc = source::source_location::current()) noexcept;

        /**@brief Using templates to allow user to use formattet logging.
         * @refitem @FormatString Is used to obtain name of calling func, file and line number as default parameter */

        template<typename... Args>
        void info(const FormatString &format, Args &&... args) {
            vinfo(format, fmt::make_format_args(args...));

        }

        void vinfo(const FormatString &format, fmt::format_args args) {
#if defined(HAS_SOURCE_LOCATION) || defined(HAS_SOURCE_LOCATION_EXPERIMENTAL)

            const auto &loc = format.loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.str, args);

            std::string preText = fmt::format(" {}:{}: ", loc.file_name(), loc.line());
            preText.append(s);


            std::size_t found = preText.find_last_of('/');
            std::string msg = preText.substr(found + 1);

            msg = msg.insert(0, (std::to_string(frameNumber) + "  ")) ;
            _info(msg.c_str());
#else
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.str, args);
            s = s.insert(0, (std::to_string(frameNumber) + "  ")) ;
            _info(s.c_str());
#endif

        }


        // Interface for Trace log
        void trace(const char *text) throw();

        void trace(std::string &text) throw();

        void trace(std::ostringstream &stream) throw();

        // Interface for Debug log
        void debug(const char *text) throw();

        void debug(std::string &text) throw();

        void debug(std::ostringstream &stream) throw();

        // Error and Alarm log must be always enable
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
        void _info(const char *text) throw();

        void logIntoFile(std::string &data);

        void logOnConsole(std::string &data);

        Logger(const Logger &obj) {}

        void operator=(const Logger &obj) {}

    private:
        static Logger *m_Instance;
        std::ofstream m_File;

        std::mutex m_Mutex{};

        LogLevel m_LogLevel;
        LogType m_LogType;

    };

} // End of namespace

#endif // End of _LOGGER_H_

