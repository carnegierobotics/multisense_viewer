cmake_minimum_required(VERSION 3.15)
project(Logger LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

add_executable(logger main.cpp Logger.cpp Logger.h)
target_link_libraries(logger -pthread)
