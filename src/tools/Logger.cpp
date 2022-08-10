// C++ Header File(s)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

// Code Specific Header Files(s)
#include "Logger.h"
#include <stdarg.h>
using namespace std;
namespace Log {

    Logger *Logger::m_Instance = nullptr;

// Log file name. File name should be change from here only
    const string logFileName = "logger.log";

    Logger::Logger() {
        m_File.open(logFileName.c_str(), ios::out | ios::app);
        m_LogLevel = LOG_LEVEL_TRACE;
        m_LogType = FILE_LOG;
        /*
        int ret = 0;
        ret = pthread_mutexattr_settype(&m_Attr, PTHREAD_MUTEX_ERRORCHECK_NP);
        if (ret != 0) {
            printf("Logger::Logger() -- Mutex attribute not initialize!!\n");
            exit(0);
        }
        ret = pthread_mutex_init(&m_Mutex, &m_Attr);
        if (ret != 0) {
            printf("Logger::Logger() -- Mutex not initialize!!\n");
            exit(0);
        }
        */
    }

    Logger::~Logger() {
        m_File.close();
        //pthread_mutexattr_destroy(&m_Attr);
        //pthread_mutex_destroy(&m_Mutex);
    }

    Logger *Logger::getInstance() throw() {
        if (m_Instance == 0) {
            m_Instance = new Logger();
        }
        return m_Instance;
    }

    void Logger::lock() {
        //pthread_mutex_lock(&m_Mutex);
    }

    void Logger::unlock() {
        //pthread_mutex_unlock(&m_Mutex);
    }

    void Logger::logIntoFile(std::string &data) {
        lock();
        m_File << getCurrentTime() << "  " << data << endl;
        unlock();
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
    void Logger::_error(const char *text) throw() {
        string data;
        data.append("[ERROR]: ");
        data.append(text);

        // ERROR must be capture
        if (m_LogType == FILE_LOG) {
            logIntoFile(data);
        } else if (m_LogType == CONSOLE) {
            logOnConsole(data);
        }
    }

    void Logger::error(std::string &text) throw() {
        _error(text.data());
    }

    void Logger::error(std::ostringstream &stream) throw() {
        string text = stream.str();
        _error(text.data());
    }

    void Logger::error(const char *fmt, ...)
    {
        // determine required buffer size
        va_list args;
        va_start(args, fmt);
        int len = vsnprintf(NULL, 0, fmt, args);
        va_end(args);
        if(len < 0) return;

        // format message
        std::vector<char> msg;
        msg.resize(len + 1);
        //char msg[len + 1]; // or use heap allocation if implementation doesn't support VLAs
        va_start(args, fmt);
        vsnprintf(msg.data(), len + 1, fmt, args);
        va_end(args);

        // call myFunction
        _error(msg.data());
    }

// Interface for Alarm Log 
    void Logger::alarm(const char *text) throw() {
        string data;
        data.append("[ALARM]: ");
        data.append(text);

        // ALARM must be capture
        if (m_LogType == FILE_LOG) {
            logIntoFile(data);
        } else if (m_LogType == CONSOLE) {
            logOnConsole(data);
        }
    }

    void Logger::alarm(std::string &text) throw() {
        alarm(text.data());
    }

    void Logger::alarm(std::ostringstream &stream) throw() {
        string text = stream.str();
        alarm(text.data());
    }

// Interface for Always Log 
    void Logger::always(const char *text) throw() {
        string data;
        data.append("[ALWAYS]: ");
        data.append(text);

        // No check for ALWAYS logs
        if (m_LogType == FILE_LOG) {
            logIntoFile(data);
        } else if (m_LogType == CONSOLE) {
            logOnConsole(data);
        }
    }

    void Logger::always(std::string &text) throw() {
        always(text.data());
    }

    void Logger::always(std::ostringstream &stream) throw() {
        string text = stream.str();
        always(text.data());
    }

// Interface for Buffer Log 
    void Logger::buffer(const char *text) throw() {
        // Buffer is the special case. So don't add log level
        // and timestamp in the buffer message. Just log the raw bytes.
        if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_BUFFER)) {
            lock();
            m_File << text << endl;
            unlock();
        } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_BUFFER)) {
            cout << text << endl;
        }
    }

    void Logger::buffer(std::string &text) throw() {
        buffer(text.data());
    }

    void Logger::buffer(std::ostringstream &stream) throw() {
        string text = stream.str();
        buffer(text.data());
    }

// Interface for Info Log
    void Logger::_info(const char *text) throw() {
        string data;
        data.append("[INFO]: ");
        data.append(text);

        if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_INFO)) {
            logIntoFile(data);
        } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_INFO)) {
            logOnConsole(data);
        }
    }

    void Logger::info(std::string &text, const std::source_location& loc) throw() {
        _info(text.data());
    }
/*
    void Logger::info(std::ostringstream &stream) throw() {
        string text = stream.str();
        _info(text.data());
    }


    void Logger::info(const char *fmt, ...)
    {

        // determine required buffer size
        va_list args;
        va_start(args, fmt);
        int len = vsnprintf(NULL, 0, fmt, args);
        va_end(args);
        if(len < 0) return;

        // format message
        std::vector<char> msg;
        msg.resize(len + 1);
        //char msg[len + 1]; // or use heap allocation if implementation doesn't support VLAs
        va_start(args, fmt);
        vsnprintf(msg.data(), len + 1, fmt, args);
        va_end(args);

        // call myFunction
        _info(msg.data());
    }

 */
// Interface for Trace Log
    void Logger::trace(const char *text) throw() {
        string data;
        data.append("[TRACE]: ");
        data.append(text);

        if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
            logIntoFile(data);
        } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
            logOnConsole(data);
        }
    }

    void Logger::trace(std::string &text) throw() {
        trace(text.data());
    }

    void Logger::trace(std::ostringstream &stream) throw() {
        string text = stream.str();
        trace(text.data());
    }

// Interface for Debug Log
    void Logger::debug(const char *text) throw() {
        string data;
        data.append("[DEBUG]: ");
        data.append(text);

        if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_DEBUG)) {
            logIntoFile(data);
        } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_DEBUG)) {
            logOnConsole(data);
        }
    }

    void Logger::debug(std::string &text) throw() {
        debug(text.data());
    }

    void Logger::debug(std::ostringstream &stream) throw() {
        string text = stream.str();
        debug(text.data());
    }

// Interfaces to control log levels
    void Logger::updateLogLevel(LogLevel logLevel) {
        m_LogLevel = logLevel;
    }

// Enable all log levels
    void Logger::enaleLog() {
        m_LogLevel = ENABLE_LOG;
    }

// Disable all log levels, except error and alarm
    void Logger::disableLog() {
        m_LogLevel = DISABLE_LOG;
    }

// Interfaces to control log Types
    void Logger::updateLogType(LogType logType) {
        m_LogType = logType;
    }

    void Logger::enableConsoleLogging() {
        m_LogType = CONSOLE;
    }


void Logger::enableFileLogging()
{
   m_LogType = FILE_LOG ;
}

};