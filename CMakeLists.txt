project(NEURAL)
cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

set(INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")

add_library(neural
  src/Neuron.cpp
  src/Layer.cpp
  src/Network.cpp
  )

target_include_directories(neural PUBLIC ${INCLUDE_DIRS})
