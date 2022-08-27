######################
# set up the project #
######################
cmake_minimum_required(VERSION 2.8.3)
project(ipopt_eigen_wrapper)

# Initial message
message(STATUS "Compiling locomotionutils library")


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


find_package(PkgConfig)
pkg_check_modules(EIGEN3 eigen3>=3.3.2 REQUIRED)


############################

find_package(IPOPT REQUIRED)



# Include directories
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})



message(STATUS "Including IPOPT features")

include_directories(${IPOPT_INCLUDE_DIRS})#must be after find_package

message( ${IPOPT_INCLUDE_DIRS})

set(IPOPT_HEADERS ./include/TNLP_EigenDense.hpp ./include/hs071_nlp.hpp)

set(IPOPT_SOURCES ./src/TNLP_EigenDense.cpp ./src/TNLP_EigenDenseExample.cpp ./src/hs071_nlp.cpp)

add_library(${PROJECT_NAME} SHARED ${IPOPT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${IPOPT_LIBRARIES})

add_executable(customtest src/test_TNLP_EigenDense.cpp)
target_link_libraries(customtest ${PROJECT_NAME}  ${IPOPT_LIBRARIES})

install(TARGETS  customtest
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


