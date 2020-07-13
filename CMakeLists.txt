CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(rts)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors -g")

# execute_process(
#     COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
#     OUTPUT_VARIABLE TorchPath
# )
# set(CMAKE_PREFIX_PATH ${TorchPath})
# find_package(Torch REQUIRED)
# set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")
# get and append paths for finding dep

execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so")

add_subdirectory(third_party/pybind11)
add_subdirectory(tube)
include_directories(tube/src_cpp)

add_subdirectory(game)

add_subdirectory(third_party)
