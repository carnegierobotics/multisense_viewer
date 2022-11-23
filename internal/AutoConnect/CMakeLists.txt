cmake_minimum_required(VERSION 3.15)
project(AutoConnect LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

include_directories("include")

option(BUILD_STANDALONE "build this executable without viewer" ON)

if (BUILD_STANDALONE)
    set(LIBMULTISENSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/LibMultiSense/source/LibMultiSense)
    include_directories(${LIBMULTISENSE_DIR})
    add_subdirectory(${LIBMULTISENSE_DIR})
else()
    message("Building integrated version")
endif()

if (WIN32)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    endif ()

    # Include WinPcap headers

    # Link WinPcap library
    add_library(winpcap STATIC IMPORTED GLOBAL)
    set_property(TARGET winpcap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(winpcap PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
    IMPORTED_LOCATION_RELEASE  "${CMAKE_CURRENT_SOURCE_DIR}/wpcap.lib")

    add_library(LibAutoConnect STATIC AutoConnect.h AutoConnectWindows.cpp AutoConnectWindows.h "WinRegEditor.h")
    target_link_libraries(LibAutoConnect MultiSense winpcap iphlpapi )

    if (MSVC)
        if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message("\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set 'GLFW-CMake-starter' as StartUp Project in Visual Studio.\n")
        else ()
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT AutoConnect)
        endif ()
    endif ()
endif ()

if(UNIX)
    ## Linux
    add_library(LibAutoConnect STATIC src/AutoConnectLinux.cpp)
    target_link_libraries(LibAutoConnect -lpthread -ltbb MultiSense)

    add_executable(AutoConnect src/main.cpp)
    target_link_libraries(AutoConnect LibAutoConnect)
endif ()