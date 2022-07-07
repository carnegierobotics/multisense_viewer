cmake_minimum_required(VERSION 3.15)
project(cpp_boilerplate LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

set(MULTISENSE_SRC ../external/LibMultiSense/source/LibMultiSense)
add_subdirectory(${MULTISENSE_SRC}  [EXCLUDE_FROM_ALL])

find_package( OpenCV REQUIRED )

add_executable(listAdapter ListAdapters.cpp )
target_link_libraries(listAdapter MultiSense)

add_executable(broadcast_listener BroadcastListener.cpp )

add_executable(igmpsniffer igmp_sniffer.cpp )
target_link_libraries(igmpsniffer -ltbb)

add_executable(packet_sniffer packet_sniffer.cpp )
target_link_libraries(packet_sniffer -ltbb)

set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)

add_executable(decode_mp4 decode_mp4.cpp)
target_link_libraries(decode_mp4 PRIVATE -lavformat -lavcodec -lavutil  -lz -lm -lpthread ${OpenCV_LIBS})

add_executable(decode decode.cpp)
target_link_libraries(decode PRIVATE -lavcodec -lavutil -lswscale ${OpenCV_LIBS})