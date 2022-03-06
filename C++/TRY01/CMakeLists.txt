cmake_minimum_required(VERSION 3.0.0)
project(start VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(start main.cpp)
# Where to find CMake modules and OpenCV
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.0_5")
set(OpenCV_INCLUDE_DIRS "/usr/local/Cellar/opencv/4.5.0_5/include")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(learn_cpp ${SOURCE_FILES})

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
# or add all
include_directories(${OpenCV_INCLUDE_DIRS})

# linking
#target_link_libraries(learn_cpp ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
