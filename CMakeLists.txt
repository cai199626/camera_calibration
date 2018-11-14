cmake_minimum_required(VERSION 3.12)
project(camera_calibration)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3.0 REQUIRED)


add_executable(camera_calibration main.cpp)
target_link_libraries(camera_calibration ${OpenCV_LIBS})