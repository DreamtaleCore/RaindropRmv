cmake_minimum_required(VERSION 2.8)


project(Edge)
find_package(OpenCV REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Werror")
SET(CMAKE_CXX_FLAGS "-std=c++14 -O3")

include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)

add_executable(Edge ${DIR_SRCS})
target_link_libraries(Edge ${OpenCV_LIBS})
