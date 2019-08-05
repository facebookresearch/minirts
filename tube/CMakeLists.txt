cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")


find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)


# lib for other c++ programs
add_library(_tube src_cpp/data_channel.cc)
target_include_directories(_tube PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_tube PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_tube PUBLIC ${TORCH_LIBRARIES})

# tests
add_executable(test_data_channel src_cpp/test/test_data_channel.cc)
target_link_libraries(test_data_channel _tube)

# pybind lib
pybind11_add_module(tube src_cpp/pybind.cc)
target_link_libraries(tube PUBLIC _tube)
