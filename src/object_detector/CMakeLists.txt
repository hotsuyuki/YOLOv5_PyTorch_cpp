cmake_minimum_required(VERSION 3.12)

project(yolov5-object-detector-lib
  VERSION 1.0.0
  DESCRIPTION "YOLOv5 ObjectDetector library"
  LANGUAGES CXX
)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
  message(STATUS "OpenCV library status:")
  message(STATUS "  version: ${OpenCV_VERSION}")
  message(STATUS "  include path: ${OpenCV_INCLUDE_DIRS}")
else ()
  message(FATAL_ERROR "Could not find OpenCV")
endif ()

set(Torch_DIR ${CMAKE_SOURCE_DIR}/libtorch_v1-6-0/share/cmake/Torch/)
find_package(Torch PATHS ${Torch_DIR} REQUIRED)
if (TORCH_FOUND)
  message(STATUS "Torch library status:")
  message(STATUS "  version: ${Torch_VERSION}")
  message(STATUS "  include path: ${TORCH_INCLUDE_DIRS}")
else ()
  message(FATAL_ERROR "Could not find Torch")
endif ()

add_library(object_detector STATIC
  ${PROJECT_SOURCE_DIR}/src/object_detector.cpp
)
target_compile_features(object_detector PRIVATE cxx_std_17)

target_include_directories(object_detector PUBLIC
  ${PROJECT_SOURCE_DIR}/include/
)

target_link_libraries(object_detector
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)
