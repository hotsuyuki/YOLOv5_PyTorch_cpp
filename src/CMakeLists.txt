cmake_minimum_required(VERSION 3.12)

project(YOLOv5_PyTorch_cpp
  VERSION 1.0.0
  DESCRIPTION "Ultralytics LLC's YOLOv5 with PyTorch c++ API"
  LANGUAGES CXX 
)

set(Torch_DIR ${CMAKE_SOURCE_DIR}/libtorch_v1-6-0/share/cmake/Torch/)
find_package(Torch PATHS ${Torch_DIR} REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/object_detector/)

add_executable(main ${CMAKE_SOURCE_DIR}/main.cpp)
target_compile_features(main PRIVATE cxx_std_17)

target_include_directories(main PRIVATE
  ${CMAKE_SOURCE_DIR}/cxxopts/
)

target_link_libraries(main
  object_detector
)