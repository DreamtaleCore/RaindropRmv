cmake_minimum_required(VERSION 2.8)


project(Test)
find_package(OpenCV REQUIRED)
find_library(JSONCPP_LIBS jsoncpp)
find_library(ARMA_LIBS armadillo)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic ")
SET(CMAKE_CXX_FLAGS "-std=c++14 -O3")

include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)

add_executable(Test ${DIR_SRCS})
target_link_libraries(Test ${OpenCV_LIBS}
                           ${JSONCPP_LIBS}
                           ${ARMA_LIBS})

