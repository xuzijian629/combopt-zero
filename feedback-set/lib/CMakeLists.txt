cmake_minimum_required(VERSION 3.11)
project(lib)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS "-Wall -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_DEBUG -ftrapv")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/lib source)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src source)

add_library(_feedback_set SHARED ${source})
target_link_libraries(_feedback_set "${TORCH_LIBRARIES}")
