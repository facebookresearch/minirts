CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(rts)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors -g")

execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
)
set(CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

add_subdirectory(third_party/pybind11)
add_subdirectory(tube)
include_directories(tube/src_cpp)

add_subdirectory(game)

add_subdirectory(third_party)
