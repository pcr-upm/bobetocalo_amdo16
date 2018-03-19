#------------------------------------------------------------------------------
# Test face headpose algorithms
#------------------------------------------------------------------------------
project(bobetocalo_amdo16)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)

#-- Setup required libraries
find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS FOUND_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11")
else()
  message(STATUS NOT_FOUND_OPENMP)
endif()

#-- Find include files
set(face_headpose_bobetocalo_amdo16_include
  ${faces_framework_include}
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_headpose_bobetocalo_amdo16_src
  ${faces_framework_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceHeadPoseForest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/HeadPoseSample.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/ImageSample.cpp
)

set(face_headpose_bobetocalo_amdo16_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_headpose_bobetocalo_amdo16_test.cpp
)

set(face_headpose_bobetocalo_amdo16_libs
  ${faces_framework_libs}
)

#-- Setup CMake to run tests
enable_testing()

include_directories(${face_headpose_bobetocalo_amdo16_include})

foreach(test ${face_headpose_bobetocalo_amdo16_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_headpose_bobetocalo_amdo16_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_headpose_bobetocalo_amdo16_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name})
endforeach()
