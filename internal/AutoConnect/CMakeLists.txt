cmake_minimum_required(VERSION 3.15)
project(AutoConnect LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

option(LIBMULTISENSE_PATH "Path to LibMultiSense Source directory" None)
option(LIBMULTISENSE_SKIP "Skip looking for LibMultiSense" False)
option(BUILD_AUTOCONNECT_LIB FALSE)

set(LIBMULTISENSE_PATH "../../external/LibMultiSense")
set(BUILD_AUTOCONNECT_LIB True)

if(NOT DEFINED LIBMULTISENSE_PATH)
    message(FATAL_ERROR "You must set LIBMULTISENSE_PATH environment variable")
elseif(LIBMULTISENSE_SKIP STREQUAL True)
    # Do nothing. Use inherited value for MULTISENSE_SRC
    set(BUILD_AUTOCONNECT_LIB TRUE)
else()
    set(MULTISENSE_SRC ${LIBMULTISENSE_PATH})
    include_directories(${MULTISENSE_SRC})
    add_subdirectory(${MULTISENSE_SRC}  [EXCLUDE_FROM_ALL])
endif()


if (WIN32)

    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    endif ()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(BUILD_SHARED_LIBS TRUE)
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
    endif ()

    include_directories("./include")

    add_library(winpcap STATIC IMPORTED GLOBAL)
    set_property(TARGET winpcap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(winpcap PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
    IMPORTED_LOCATION_RELEASE  "${CMAKE_CURRENT_SOURCE_DIR}/wpcap.lib")

    message(CURRENT PATH: ${CMAKE_CURRENT_SOURCE_DIR} | ${BUILD_AUTOCONNECT_LIB})

    if (${BUILD_AUTOCONNECT_LIB})
        add_library(LibAutoConnect STATIC AutoConnect.h AutoConnectWindows.cpp AutoConnectWindows.h)
        target_link_libraries(LibAutoConnect MultiSense winpcap iphlpapi)
    else()
        add_executable(AutoConnect WIN32 AutoConnect.h AutoConnectWindows.cpp AutoConnectWindows.h main.cpp)
        target_link_libraries(AutoConnect MultiSense winpcap iphlpapi)
    endif()


    if (MSVC)
        if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message("\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set 'GLFW-CMake-starter' as StartUp Project in Visual Studio.\n")
        else ()
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT AutoConnect)
        endif ()
    endif ()


else () ## Linux
    if (${BUILD_AUTOCONNECT_LIB})
        add_library(LibAutoConnect AutoConnect.h AutoConnectLinux.cpp AutoConnectLinux.h)
        target_link_libraries(LibAutoConnect MultiSense -ltbb)
    else()
        add_executable(AutoConnect AutoConnect.h AutoConnectLinux.cpp AutoConnectLinux.h main.cpp)
        target_link_libraries(AutoConnect MultiSense -ltbb)
    endif()


endif ()