cmake_minimum_required(VERSION 2.8.3)
project(Gaussian CXX C)

find_package(CVT_DEBUG REQUIRED )

###########
## Build ##
###########

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

include_directories(
  include
  ${CVT_INCLUDE_DIRS}
)

ADD_EXECUTABLE( gaussian
  gaussian.cpp
)
TARGET_LINK_LIBRARIES( gaussian
  ${CVT_LIBRARIES}
)

ADD_EXECUTABLE( gaussian_online
  gaussian_online.cpp
)
TARGET_LINK_LIBRARIES( gaussian_online
  ${CVT_LIBRARIES}
)
